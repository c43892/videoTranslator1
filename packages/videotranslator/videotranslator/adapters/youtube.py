"""Bounded yt-dlp subprocess for validated public single-video URLs."""
import os
import subprocess
import sys
from pathlib import Path

from .conversation import youtube_url


def _supported_extractor(url: str) -> bool:
    from yt_dlp.extractor import gen_extractors
    return any(ie.IE_NAME != "generic" and ie.suitable(url) for ie in gen_extractors())


class YtDlpVideoImporter:
    def download(self, url: str, destination: Path, max_bytes: int) -> Path:
        canonical = youtube_url(url)
        if not _supported_extractor(canonical):
            raise ValueError("youtube_download_failed")
        command = [sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist",
            "--no-progress", "--quiet", "--no-warnings", "--socket-timeout", "20",
            "--retries", "2", "--fragment-retries", "2", "--max-filesize", str(max_bytes),
            "--match-filters", "!is_live & duration <= 14400",
            "-f", "bv*[height<=1080]+ba/b[height<=1080]", "--merge-output-format", "mp4",
            "-o", str(destination / "video.%(ext)s"), canonical]
        if os.environ.get('YTDLP_JS_RUNTIME'):
            command[3:3] = ['--js-runtimes', os.environ['YTDLP_JS_RUNTIME']]
        try:
            subprocess.run(command, check=True, timeout=900, capture_output=True)
        except (subprocess.SubprocessError, OSError) as exc:
            # Provider output may contain signed URLs; do not expose it to the client.
            raise ValueError("youtube_download_failed") from exc
        files = [p for p in destination.glob("video.*") if p.suffix in {".mp4", ".mkv", ".webm"}]
        if len(files) != 1 or not 0 < files[0].stat().st_size <= max_bytes:
            raise ValueError("youtube_download_failed")
        return files[0]
