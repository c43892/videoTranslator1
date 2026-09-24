"""Reverse registration: private workers poll HTTPS; the queue lives on the server.

No HTTP handler waits for a worker or a free slot. All changes are transactional,
and each assignment has a fencing token so old uploads cannot finish new attempts.
"""
import hashlib
import secrets
from pathlib import Path

from ..domain.conversation import Conversation
from ..domain.downloads import DownloadTask, DownloadWorker
from ..domain.models import now_ms
from ..ids import new_id


class DownloadConflict(ValueError):
    pass


class HomeDownloads:
    INTERVAL_MS = 10_000
    MAX_RETRIES = 10
    PRESENCE_MS = 30_000
    LEASE_MS = 45_000
    MAX_JOB_MS = 60 * 60 * 1000
    CAPACITY = 2

    def __init__(self, store, storage, settings, clock=now_ms):
        self.store, self.storage, self.settings, self.clock = store, storage, settings, clock

    @property
    def enabled(self):
        return self.settings.youtube_download_mode == "home-worker"

    def authenticate(self, token):
        if not self.enabled or not token:
            return None
        for allowed in self.settings.youtube_worker_tokens:
            if secrets.compare_digest(token, allowed):
                return hashlib.sha256(allowed.encode()).hexdigest()[:24]
        return None

    def enqueue(self, tx, draft):
        now = self.clock()
        if draft.download_task_id:
            previous = tx.get(DownloadTask, draft.download_task_id)
            if previous:
                previous.status = "abandoned"
                tx.put(previous, previous.task_id)
        task = DownloadTask(task_id=new_id("dl"), conversation_id=draft.conversation_id,
                            url=draft.youtube_url, created_at=now, next_check_at=now + self.INTERVAL_MS)
        tx.insert(task, task.task_id)
        draft.download_task_id = task.task_id
        draft.status, draft.lease_until, draft.error = "import_pending", 0, ""

    def get(self, task_id):
        with self.store.transaction() as tx:
            return tx.get(DownloadTask, task_id)

    def _known_workers(self):
        return {hashlib.sha256(t.encode()).hexdigest()[:24] for t in self.settings.youtube_worker_tokens}

    def _fail(self, tx, task, error):
        task.status, task.error, task.lease_until = "failed", error, 0
        tx.put(task, task.task_id)
        draft = tx.get(Conversation, task.conversation_id)
        if draft and draft.download_task_id == task.task_id and draft.status in {"import_pending", "importing"}:
            draft.status, draft.error, draft.lease_until = "import_failed", error, 0
            tx.put(draft, draft.conversation_id)

    def _tick(self, tx, now):
        known = self._known_workers()
        online = any(w.worker_id in known and now - w.last_seen < self.PRESENCE_MS
                     for w in tx.query(DownloadWorker))
        for task in tx.query(DownloadTask, where_in=("status", ["queued", "leased"])):
            if task.status == "leased":
                if task.deadline <= now:
                    self._fail(tx, task, "youtube_download_timeout")
                    continue
                if task.lease_until > now and task.worker_id in known:
                    continue
                task.status, task.worker_id, task.lease_token, task.lease_until = "queued", "", "", 0
                # The liveness window has elapsed; start ten spaced retries now.
                task.next_check_at, task.unavailable_retries = now + self.INTERVAL_MS, 0
                draft = tx.get(Conversation, task.conversation_id)
                if draft and draft.download_task_id == task.task_id:
                    draft.status = "import_pending"
                    tx.put(draft, draft.conversation_id)
            if online:
                # Busy but alive is normal queueing, never an unavailable retry.
                task.next_check_at, task.unavailable_retries = now + self.INTERVAL_MS, 0
            elif now >= task.next_check_at:
                task.unavailable_retries += 1
                task.next_check_at = now + self.INTERVAL_MS
                if task.unavailable_retries >= self.MAX_RETRIES:
                    self._fail(tx, task, "youtube_proxy_unavailable")
                    continue
            tx.put(task, task.task_id)

    def tick(self):
        with self.store.transaction() as tx:
            self._tick(tx, self.clock())
            garbage = tx.query(DownloadTask, where_in=("status", ["consumed", "abandoned"]))
        for task in garbage:
            if task.object_key:
                self.discard_result(task.task_id)

    def heartbeat(self, worker_id, active):
        now, accepted = self.clock(), []
        with self.store.transaction() as tx:
            worker = tx.get(DownloadWorker, worker_id)
            if worker is None:
                worker = DownloadWorker(worker_id=worker_id, last_seen=now)
                tx.insert(worker, worker_id)
            else:
                worker.last_seen = now
                tx.put(worker, worker_id)
            for task_id, token in active:
                task = tx.get(DownloadTask, task_id)
                if (task and task.status == "leased" and task.worker_id == worker_id
                        and secrets.compare_digest(task.lease_token, token)
                        and task.lease_until > now and task.deadline > now):
                    task.lease_until = min(now + self.LEASE_MS, task.deadline)
                    tx.put(task, task_id)
                    accepted.append(task_id)
        return {"active": accepted, "heartbeat_seconds": 5, "capacity": self.CAPACITY}

    def claim(self, worker_id):
        now = self.clock()
        with self.store.transaction() as tx:
            self._tick(tx, now)
            worker = tx.get(DownloadWorker, worker_id)
            if not worker or now - worker.last_seen >= self.PRESENCE_MS:
                raise DownloadConflict("worker_heartbeat_required")
            tasks = tx.query(DownloadTask, where_in=("status", ["queued", "leased"]), order_by="created_at")
            if sum(t.status == "leased" and t.worker_id == worker_id for t in tasks) >= self.CAPACITY:
                return None
            for task in tasks:
                if task.status != "queued":
                    continue
                draft = tx.get(Conversation, task.conversation_id)
                if not draft or draft.download_task_id != task.task_id or draft.status != "import_pending":
                    self._fail(tx, task, "youtube_download_cancelled")
                    continue
                task.status, task.worker_id = "leased", worker_id
                task.lease_token = secrets.token_urlsafe(32)
                task.lease_until, task.deadline = now + self.LEASE_MS, now + self.MAX_JOB_MS
                tx.put(task, task.task_id)
                draft.status = "importing"
                tx.put(draft, draft.conversation_id)
                return {"task_id": task.task_id, "lease_token": task.lease_token, "url": task.url,
                        "max_bytes": self.settings.max_upload_bytes, "timeout_seconds": 1800}
        return None

    def _leased(self, tx, task_id, worker_id, token):
        task = tx.get(DownloadTask, task_id)
        if (not task or task.worker_id != worker_id or not secrets.compare_digest(task.lease_token, token)
                or task.status != "leased" or task.lease_until <= self.clock() or task.deadline <= self.clock()):
            raise DownloadConflict("download_lease_lost")
        return task

    def check_upload(self, task_id, worker_id, token):
        with self.store.transaction() as tx:
            return self._leased(tx, task_id, worker_id, token)

    def complete(self, task_id, worker_id, token, path: Path):
        # A unique immutable object per request also fences simultaneous uploads.
        self.check_upload(task_id, worker_id, token)
        size = path.stat().st_size
        if not 0 < size <= self.settings.max_upload_bytes:
            raise ValueError("invalid_file_size")
        object_key = f"inputs/home-downloads/{task_id}/{new_id('file')}.mp4"
        self.storage.upload(path, object_key)
        try:
            with self.store.transaction() as tx:
                task = self._leased(tx, task_id, worker_id, token)
                task.status, task.object_key, task.size_bytes, task.lease_until = "ready", object_key, size, 0
                tx.put(task, task_id)
        except Exception:
            self.storage.delete(object_key)
            raise
        return {"completed": True}

    def fail(self, task_id, worker_id, token):
        with self.store.transaction() as tx:
            task = self._leased(tx, task_id, worker_id, token)
            self._fail(tx, task, "youtube_download_failed")

    def consume(self, task_id, directory):
        task = self.get(task_id)
        if not task or task.status != "ready":
            raise DownloadConflict("download_not_ready")
        return self.storage.download(task.object_key, directory / "video.mp4")

    def discard_result(self, task_id):
        task = self.get(task_id)
        if task and task.object_key:
            self.storage.delete(task.object_key)
            with self.store.transaction() as tx:
                task = tx.get(DownloadTask, task_id)
                task.status, task.object_key = "consumed", ""
                tx.put(task, task_id)
