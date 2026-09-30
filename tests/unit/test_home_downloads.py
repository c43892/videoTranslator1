"""Home-worker contracts: durable FIFO, two slots, availability and fenced uploads."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from videotranslator.api.app import create_app
from videotranslator.application.downloads import DownloadConflict, HomeDownloads
from videotranslator.bootstrap import build_container
from videotranslator.config import Settings
from videotranslator.docstore import SQLiteStore
from videotranslator.domain.conversation import Conversation
from videotranslator.domain.downloads import DownloadTask
from videotranslator.domain.models import Job, MediaInspectionResult

TOKEN = "home-test-" + "a" * 40
OTHER = "home-test-" + "b" * 40
URL = "https://www.youtube.com/watch?v=DlldBRDJXE4"
PORNHUB_URL = "https://www.pornhub.com/view_video.php?viewkey=ph5af5fef7c2aa7"
VIMEO_URL = "https://vimeo.com/123456789"


@pytest.fixture
def home(container):
    settings = replace(container.settings, youtube_download_mode="home-worker", youtube_worker_tokens=(TOKEN, OTHER))
    service = container.conversations
    service.settings = container.settings = settings
    broker = HomeDownloads(container.store, container.storage, settings)
    service.home_downloads = broker
    time = [1_800_000_000_000]
    broker.clock = lambda: time[0]
    return container, broker, time


def enqueue(container, key="chat-1", *, online=True):
    service = container.conversations
    with container.store.transaction() as tx:
        tx.insert(Conversation(conversation_id=key, owner_user_id="owner", youtube_url=URL,
                               source_kind="youtube", target_language="en"), key)
    if online:
        register(service.home_downloads)
    return service.prepare(key, "owner", 0)


def register(broker, token=TOKEN):
    worker = broker.authenticate(token)
    broker.heartbeat(worker, [])
    return worker


def test_new_youtube_import_is_rejected_offline_and_disconnect_keeps_retry_policy(home):
    container, broker, time = home
    with pytest.raises(ValueError, match="youtube_proxy_unavailable"):
        enqueue(container, online=False)
    with container.store.transaction() as tx:
        assert tx.query(DownloadTask) == []

    register(broker)
    draft = container.conversations.prepare("chat-1", "owner", 0)
    time[0] += broker.PRESENCE_MS
    for count in range(1, 10):
        broker.tick()
        task = broker.get(draft.download_task_id)
        assert task.unavailable_retries == count and task.status == "queued"
        # Polling more frequently must never consume extra retries.
        broker.tick()
        assert broker.get(task.task_id).unavailable_retries == count
        time[0] += 10_000
    broker.tick()
    failed = container.conversations.get(draft.conversation_id, "owner")
    assert failed.status == "import_failed" and failed.error == "youtube_proxy_unavailable"
    with pytest.raises(ValueError, match="youtube_proxy_unavailable"):
        container.conversations.prepare(draft.conversation_id, "owner", 0)
    register(broker)
    retried = container.conversations.prepare(draft.conversation_id, "owner", 0)
    assert retried.download_task_id != draft.download_task_id
    assert broker.get(retried.download_task_id).unavailable_retries == 0
    assert broker.get(draft.download_task_id).status == "abandoned"


def test_public_availability_tracks_recent_heartbeat(home):
    container, broker, time = home
    client = TestClient(create_app(container))
    assert client.get("/api/v1/chat/config").json()["youtube_available"] is False
    assert client.get("/api/v1/chat/youtube-availability").json()["available"] is False
    register(broker)
    assert client.get("/api/v1/chat/config").json()["youtube_available"] is True
    assert client.get("/api/v1/chat/youtube-availability").json()["available"] is True
    time[0] += broker.PRESENCE_MS
    assert client.get("/api/v1/chat/config").json()["youtube_available"] is False


def test_http_source_selection_is_rejected_without_agent(home):
    container, broker, _ = home
    client = TestClient(create_app(container), headers={"Authorization": "Bearer fake:chat-user"})
    draft = client.post("/api/v1/conversations", json={"locale": "en"}).json()
    path = f"/api/v1/conversations/{draft['conversation_id']}/messages"
    body = {"revision": draft["revision"], "choice": "source", "value": "youtube"}
    response = client.post(path, json=body)
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "youtube_proxy_unavailable"
    register(broker)
    assert client.post(path, json=body).status_code == 200


def test_busy_worker_queues_third_without_offline_failure(home):
    container, broker, time = home
    drafts = [enqueue(container, f"chat-{i}") for i in range(3)]
    worker = register(broker)
    first, second = broker.claim(worker), broker.claim(worker)
    assert broker.claim(worker) is None
    for _ in range(30):
        time[0] += 5_000
        broker.heartbeat(worker, [(t["task_id"], t["lease_token"]) for t in (first, second)])
        broker.tick()
    queued = broker.get(drafts[2].download_task_id)
    assert queued.status == "queued" and queued.unavailable_retries == 0
    broker.fail(first["task_id"], worker, first["lease_token"])
    assert broker.claim(worker)["task_id"] == queued.task_id


def test_parallel_claims_cannot_exceed_two_per_credential(home):
    container, broker, _ = home
    for i in range(8):
        enqueue(container, f"chat-{i}")
    worker = register(broker)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: broker.claim(worker), range(8)))
    assigned = [task for task in results if task]
    assert len(assigned) == 2
    assert len({t["task_id"] for t in assigned}) == 2


def test_reconnect_before_limit_resumes_and_clears_retry_counter(home):
    container, broker, time = home
    draft = enqueue(container)
    for _ in range(9):
        time[0] += 10_000
        broker.tick()
    worker = register(broker)
    task = broker.claim(worker)
    assert task["task_id"] == draft.download_task_id
    assert broker.get(task["task_id"]).unavailable_retries == 0


def test_expired_assignment_is_requeued_and_old_upload_rejected(home, tmp_path):
    container, broker, time = home
    draft = enqueue(container)
    worker = register(broker)
    old = broker.claim(worker)
    time[0] += 46_000
    worker = register(broker)
    new = broker.claim(worker)
    assert new["task_id"] == old["task_id"] and new["lease_token"] != old["lease_token"]
    path = tmp_path / "video.mp4"
    path.write_bytes(b"video")
    with pytest.raises(DownloadConflict):
        broker.complete(old["task_id"], worker, old["lease_token"], path)
    assert broker.get(draft.download_task_id).status == "leased"


def test_late_heartbeat_cannot_resurrect_expired_lease(home):
    container, broker, time = home
    enqueue(container)
    worker = register(broker)
    task = broker.claim(worker)
    time[0] += 46_000
    assert broker.heartbeat(worker, [(task["task_id"], task["lease_token"])])["active"] == []


def test_removed_token_is_not_considered_available(home):
    container, broker, time = home
    draft = enqueue(container)
    register(broker)
    broker.settings = replace(broker.settings, youtube_worker_tokens=(OTHER,))
    time[0] += 10_000
    broker.tick()
    assert broker.get(draft.download_task_id).unavailable_retries == 1


def test_actual_upload_reaches_quote_without_gpu_or_charge(home, tmp_path):
    container, broker, _ = home
    class Inspector:
        def inspect(self, path):
            assert path.read_bytes() == b"fixture-video"
            return MediaInspectionResult(72_000, "72.0", "video", True)
    container.conversations.inspector = Inspector()
    draft = enqueue(container)
    worker = register(broker)
    task = broker.claim(worker)
    client = TestClient(create_app(container))
    response = client.put(f'/api/v1/download-workers/tasks/{task["task_id"]}/content',
        headers={"Authorization": "Bearer " + TOKEN, "X-Download-Lease": task["lease_token"]}, content=b"fixture-video")
    assert response.status_code == 200, response.text
    assert broker.get(task["task_id"]).status == "ready"
    container.conversations.import_one()
    done = container.conversations.get(draft.conversation_id, "owner")
    assert done.status == "draft" and done.duration_ms == 72_000
    view = container.conversations.view(done)
    assert "download_task_id" not in view and "lease_token" not in str(view)
    with container.store.transaction() as tx:
        assert tx.query(Job) == []
    assert broker.get(task["task_id"]).status == "consumed"


def test_worker_api_requires_worker_secret_not_user_auth(home):
    container, broker, _ = home
    client = TestClient(create_app(container))
    for auth in ("", "Bearer fake:owner", "Bearer wrong"):
        assert client.post('/api/v1/download-workers/claim', headers={"Authorization": auth}).status_code == 401
    response = client.post('/api/v1/download-workers/heartbeat', headers={"Authorization": "Bearer " + TOKEN}, json={})
    assert response.status_code == 200 and response.json()["capacity"] == 2


def test_upload_rejects_other_worker_and_wrong_size(home):
    container, broker, _ = home
    enqueue(container)
    worker = register(broker)
    task = broker.claim(worker)
    client = TestClient(create_app(container))
    url = f'/api/v1/download-workers/tasks/{task["task_id"]}/content'
    headers = {"Authorization": "Bearer " + OTHER, "X-Download-Lease": task["lease_token"]}
    assert client.put(url, headers=headers, content=b"video").status_code == 409
    headers["Authorization"] = "Bearer " + TOKEN
    headers["Content-Length"] = str(container.settings.max_upload_bytes + 1)
    assert client.put(url, headers=headers, content=b"video").status_code == 413
    headers["Content-Length"] = "2"
    assert client.put(url, headers=headers, content=b"video").status_code == 413


def test_inflight_upload_is_fenced_if_lease_changes_during_storage_write(home, tmp_path):
    container, broker, time = home
    enqueue(container)
    worker = register(broker)
    task = broker.claim(worker)
    path = tmp_path / "media.mp4"
    path.write_bytes(b"data")
    original = container.storage.upload
    objects = []
    def upload(path, key):
        objects.append(key)
        original(path, key)
        time[0] += 46_000
        broker.tick()
    container.storage.upload = upload
    with pytest.raises(DownloadConflict):
        broker.complete(task["task_id"], worker, task["lease_token"], path)
    assert not container.storage.exists(objects[0])


def test_heartbeat_cannot_extend_job_forever(home):
    container, broker, time = home
    draft = enqueue(container)
    worker = register(broker)
    task = broker.claim(worker)
    time[0] += broker.MAX_JOB_MS
    broker.tick()
    assert broker.get(task["task_id"]).error == "youtube_download_timeout"
    assert container.conversations.get(draft.conversation_id, "owner").status == "import_failed"


def test_queue_survives_sqlite_restart(tmp_path):
    settings = Settings(profile="test", store_path=str(tmp_path / "store.db"),
        local_storage_dir=str(tmp_path / "objects"), youtube_download_mode="home-worker", youtube_worker_tokens=(TOKEN,))
    first = build_container(settings)
    draft = enqueue(first)
    second = build_container(settings)
    broker = second.conversations.home_downloads
    worker = register(broker)
    assert broker.claim(worker)["task_id"] == draft.download_task_id


def test_worker_configuration_rejects_remote_http_and_url_credentials(tmp_path):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("home_worker", root / "deploy/home-download-worker/worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import json
    path = tmp_path / "config.json"
    for server in ("http://public.example", "https://user:password@example.com", "https://example.com/?token=secret"):
        path.write_text(json.dumps({"server_url": server, "token": TOKEN}))
        with pytest.raises(ValueError):
            module.load_config(path)
    for bad_url in ("file:///etc/passwd", "https://127.0.0.1/video", "https://192.168.1.1/video",
                    "https://media.internal/video"):
        with pytest.raises(ValueError):
            module.canonical_url(bad_url)
    assert module.canonical_url(URL) == URL
    assert module.canonical_url(PORNHUB_URL) == PORNHUB_URL
    assert module.canonical_url(VIMEO_URL) == VIMEO_URL
    assert module.supported_extractor(VIMEO_URL)
    assert not module.supported_extractor("https://example.com/video/1")
