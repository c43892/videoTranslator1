"""Portable Windows download agent: outbound HTTPS only, two bounded slots.

Run with Python 3.12+, yt-dlp[default], httpx, Node and FFmpeg available.
Credentials remain in config.json; never pass them in command-line arguments.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import ipaddress
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlsplit

import httpx

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger("home-download-worker")
CAPACITY = 2
_EXTRACTORS = None


def canonical_url(value):
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid_youtube_url") from exc
    if parsed.scheme != "https" or parsed.username or parsed.password or port or parsed.fragment:
        raise ValueError("invalid_youtube_url")
    host = (parsed.hostname or "").lower()
    if (host == "www.youtube.com" and parsed.path == "/watch"
            and re.fullmatch(r"v=[A-Za-z0-9_-]{11}", parsed.query)):
        return value
    if host == "www.pornhub.com" and parsed.path == "/view_video.php":
        keys = parse_qs(parsed.query).get("viewkey", [])
        if len(keys) == 1 and re.fullmatch(r"[A-Za-z0-9]+", keys[0]):
            return f"https://www.pornhub.com/view_video.php?viewkey={keys[0]}"
    if not host or "." not in host or host == "localhost" or host.endswith((".localhost", ".local", ".lan", ".internal")):
        raise ValueError("invalid_youtube_url")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError("invalid_youtube_url")
    return parsed._replace(fragment="").geturl()


def supported_extractor(url):
    """Reject the generic extractor so arbitrary sites cannot reach a home LAN."""
    global _EXTRACTORS
    if _EXTRACTORS is None:
        from yt_dlp.extractor import gen_extractors
        _EXTRACTORS = tuple(ie for ie in gen_extractors() if ie.IE_NAME != "generic")
    return any(ie.suitable(url) for ie in _EXTRACTORS)


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    url = urlsplit(config["server_url"])
    local = url.hostname in {"127.0.0.1", "localhost", "::1"}
    if (url.scheme != "https" and not (local and url.scheme == "http")) or not url.hostname:
        raise ValueError("server_url_requires_https")
    if url.username or url.password or url.query or url.fragment or url.path not in {"", "/"}:
        raise ValueError("server_url_must_be_an_origin")
    if len(config.get("token", "")) < 32:
        raise ValueError("token_too_short")
    config["server_url"] = config["server_url"].rstrip("/")
    return config


def terminate(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        import signal
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=10)


def download(task, directory, cancelled, config):
    url = canonical_url(task["url"])
    if not supported_extractor(url):
        raise ValueError("unsupported_video_url")
    limit = min(int(task["max_bytes"]), 2 * 1024**3)
    ffmpeg = config.get("ffmpeg_path") or shutil.which("ffmpeg")
    node = config.get("node_path") or shutil.which("node")
    if not ffmpeg or not node:
        raise ValueError("missing_ffmpeg_or_node")
    command = [sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist",
               "--no-progress", "--quiet", "--no-warnings", "--socket-timeout", "20",
               "--retries", "2", "--fragment-retries", "2", "--max-filesize", str(limit),
               "--match-filters", "!is_live & duration <= 14400", "--js-runtimes", f"node:{node}",
               "--ffmpeg-location", str(Path(ffmpeg).parent),
               "-f", "bv*[height<=1080]+ba/b[height<=1080]", "--merge-output-format", "mp4",
               "-o", str(directory / "video.%(ext)s"), url]
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
    # yt-dlp errors can contain signed media URLs. Do not persist raw output.
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)
    deadline = time.monotonic() + min(int(task.get("timeout_seconds", 1800)), 1800)
    try:
        while process.poll() is None:
            if cancelled.wait(0.5) or time.monotonic() >= deadline:
                raise RuntimeError("download_cancelled_or_timed_out")
            # Separate A/V streams and merging can temporarily need ~2x final size.
            size = 0
            for path in directory.iterdir():
                with contextlib.suppress(FileNotFoundError):
                    if path.is_file():
                        size += path.stat().st_size
            if size > 3 * limit:
                raise ValueError("download_disk_limit")
        if process.returncode:
            raise RuntimeError("youtube_download_failed")
    finally:
        terminate(process)
    files = [p for p in directory.glob("video.*") if p.suffix in {".mp4", ".mkv", ".webm"}]
    if len(files) != 1 or not 0 < files[0].stat().st_size <= limit:
        raise ValueError("invalid_download_size")
    return files[0]


class Agent:
    def __init__(self, config, root=ROOT):
        self.config, self.root = config, root
        self.base = config["server_url"] + "/api/v1/download-workers"
        self.headers = {"Authorization": "Bearer " + config["token"]}
        self.active = {}
        self.stopping = threading.Event()
        self.last_heartbeat = 0.0

    def request(self, endpoint, payload):
        with httpx.Client(timeout=10, follow_redirects=False, trust_env=False) as client:
            response = client.post(self.base + endpoint, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()

    def process(self, task, cancelled):
        task_id = task["task_id"]
        lease = {"task_id": task_id, "lease_token": task["lease_token"]}
        try:
            cache = self.root / "work"
            cache.mkdir(exist_ok=True)
            # Bound temporary storage for two slots; fail cleanly before filling a home disk.
            if shutil.disk_usage(cache).free < 6 * int(task["max_bytes"]):
                raise RuntimeError("insufficient_disk_space")
            with tempfile.TemporaryDirectory(prefix="download-", dir=cache) as directory:
                path = download(task, Path(directory), cancelled, self.config)

                def chunks():
                    with path.open("rb") as source:
                        while chunk := source.read(1024 * 1024):
                            if cancelled.is_set():
                                raise RuntimeError("lease_lost")
                            yield chunk

                with httpx.Client(timeout=httpx.Timeout(1800, connect=15), follow_redirects=False,
                                  trust_env=False) as client:
                    response = client.put(self.base + f"/tasks/{task_id}/content", content=chunks(),
                        headers={**self.headers, "X-Download-Lease": task["lease_token"],
                                 "Content-Type": "video/mp4", "Content-Length": str(path.stat().st_size)})
                    response.raise_for_status()
                LOG.info("Completed task %s", task_id)
        except Exception as exc:
            # Never log exception messages, headers, credentials, or signed URLs.
            safe_codes = {"insufficient_disk_space", "missing_ffmpeg_or_node", "invalid_youtube_url",
                          "unsupported_video_url",
                          "download_disk_limit", "download_cancelled_or_timed_out", "youtube_download_failed",
                          "invalid_download_size", "lease_lost"}
            code = str(exc) if str(exc) in safe_codes else type(exc).__name__
            LOG.warning("Task %s failed (%s)", task_id, code)
            if not cancelled.is_set():
                with contextlib.suppress(Exception):
                    self.request("/fail", lease)

    def run(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=CAPACITY) as pool:
            try:
                while not self.stopping.is_set() and not (self.root / "STOP").exists():
                    self.active = {k: v for k, v in self.active.items() if not v["future"].done()}
                    try:
                        if time.monotonic() - self.last_heartbeat >= 5:
                            active = [{"task_id": k, "lease_token": v["task"]["lease_token"]}
                                      for k, v in self.active.items()]
                            heartbeat = self.request("/heartbeat", {"active": active})
                            if not self.last_heartbeat:
                                LOG.info("Connected to VideoTranslator; capacity %s", heartbeat["capacity"])
                            self.last_heartbeat = time.monotonic()
                            status_path = self.root / "worker-status.json"
                            status_temp = self.root / "worker-status.tmp"
                            status_temp.write_text(json.dumps({"last_heartbeat_unix": time.time(),
                                "active_tasks": len(heartbeat["active"]), "capacity": heartbeat["capacity"]}))
                            status_temp.replace(status_path)
                            for key, item in self.active.items():
                                if key not in heartbeat["active"]:
                                    item["cancelled"].set()
                        while len(self.active) < CAPACITY:
                            task = self.request("/claim", {})["task"]
                            if not task:
                                break
                            cancelled = threading.Event()
                            self.active[task["task_id"]] = {"task": task, "cancelled": cancelled,
                                "future": pool.submit(self.process, task, cancelled)}
                            LOG.info("Claimed task %s", task["task_id"])
                    except Exception as exc:
                        LOG.warning("Server unavailable (%s); reconnecting", type(exc).__name__)
                        if time.monotonic() - self.last_heartbeat >= 30:
                            for item in self.active.values():
                                item["cancelled"].set()
                    self.stopping.wait(2)
            finally:
                for item in self.active.values():
                    item["cancelled"].set()


def main():
    if "--check" in sys.argv:
        load_config(ROOT / "config.json")
        return
    handler = RotatingFileHandler(ROOT / "worker.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)
    # Prevent two instances in the same installation from accepting four jobs.
    with (ROOT / "worker.lock").open("a+b") as lock:
        lock.seek(0)
        lock.write(b"0")
        lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return
        try:
            Agent(load_config(ROOT / "config.json")).run()
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            LOG.error("Worker stopped (%s)", type(exc).__name__)
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
