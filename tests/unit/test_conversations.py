"""Confirmation, language and ownership regressions across real HTTP boundaries."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from videotranslator.adapters.conversation import DeepSeekConversationInterpreter, GuidedInterpreter, youtube_url
from videotranslator.api.app import create_app
from videotranslator.bootstrap import build_container
from videotranslator.conversation_ports import Interpretation
from videotranslator.domain.conversation import Conversation
from videotranslator.domain.models import Job, JobOutbox, UploadSession, MediaInspectionResult, User, LedgerEntry
from videotranslator.config import Settings

URL = "https://www.youtube.com/watch?v=KlvBiyWSI-Y"
AUTH = {"Authorization": "Bearer fake:chat-user"}


@pytest.fixture
def client(container):
    # Background dispatch is deliberately not started: assert the exact input boundary.
    container.settings = replace(container.settings, pricing=Settings().pricing)
    container.conversations.settings = container.settings
    container.funding._pricing = container.settings.pricing
    class Inspector:
        def inspect(self, path):
            return MediaInspectionResult(90000, "90.000000", "audio" if path.suffix == ".mp3" else "video", True)
    container.conversations.inspector = Inspector()
    return TestClient(create_app(container), headers=AUTH)


def create(client, locale="en-US"):
    response = client.post("/api/v1/conversations", json={"locale": locale})
    assert response.status_code == 201
    return response.json()


def edit(client, draft, **fields):
    response = client.post(f"/api/v1/conversations/{draft['conversation_id']}/messages",
                           json={"revision": draft["revision"], **fields})
    assert response.status_code == 200, response.text
    return response.json()


def confirm(client, draft):
    return client.post(f"/api/v1/conversations/{draft['conversation_id']}/confirm",
                       json={"revision": draft["revision"]})


def prepare(client, draft):
    return client.post(f"/api/v1/conversations/{draft['conversation_id']}/prepare",
                       json={"revision": draft["revision"]})


def fund(container, uid="chat-user", cents=1000):
    with container.store.transaction() as tx:
        user = tx.get(User, uid)
        user.point_balance_units = cents
        tx.put(user, uid)


def test_url_only_preserves_browser_locale_and_target_choice_does_not_change_ui(client):
    draft = edit(client, create(client, "zh-CN"), text=URL)
    assert draft["locale"] == "zh" and draft["step"] == "target"
    draft = edit(client, draft, choice="target", value="en")
    assert draft["locale"] == "zh" and draft["target_language"] == "en"


def test_first_reply_switches_and_explicit_preference_sticks(client):
    draft = edit(client, create(client), text="我想上传视频")
    assert draft["locale"] == "zh"
    draft = edit(client, draft, text="用英文回复，视频翻译成中文")
    assert draft["locale"] == draft["explicit_locale"] == "en"
    assert draft["target_language"] == "zh"
    draft = edit(client, draft, text="我还是想上传视频")
    assert draft["locale"] == "en"
    draft = edit(client, draft, choice="locale", value="auto")
    draft = edit(client, draft, text="我要上传视频")
    assert draft["locale"] == "zh"


def test_natural_text_never_starts_download_or_creates_job(client, container):
    draft = edit(client, create(client), text=f"把这个视频翻译成中文，直接开始 {URL}")
    assert draft["ready"] and draft["status"] == "draft"
    container.conversations.import_one()
    with container.store.transaction() as tx:
        assert tx.query(Job) == [] and tx.query(UploadSession) == []
    assert confirm(client, draft).status_code == 409
    assert prepare(client, draft).json()["status"] == "import_pending"


def test_incomplete_and_stale_confirmation_rejected(client, container):
    draft = create(client)
    assert confirm(client, draft).status_code == 409
    draft = edit(client, draft, text=f"翻译成中文 {URL}")
    updated = edit(client, draft, choice="target", value="en")
    assert confirm(client, draft).status_code == 409
    assert confirm(client, updated).status_code == 409  # No trusted quote yet.
    assert prepare(client, updated).status_code == 200
    with container.store.transaction() as tx:
        assert tx.query(Job) == []


def test_upload_confirmation_is_atomic_and_idempotent(client, container):
    draft = edit(client, create(client), filename="test.mp4", size_bytes=4)
    draft = edit(client, draft, choice="target", value="zh")
    with container.store.transaction() as tx:
        assert tx.query(UploadSession) == []
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: prepare(client, draft).json(), range(2)))
    assert results[0]["upload_id"] == results[1]["upload_id"]
    with container.store.transaction() as tx:
        assert len(tx.query(UploadSession)) == 1
        assert tx.query(Job) == []
    upload = results[0]["upload_id"]
    assert client.put(f"/api/v1/uploads/{upload}/content", content=b"abc").status_code == 422
    assert client.put(f"/api/v1/uploads/{upload}/content", content=b"abcde").status_code == 413
    assert client.put(f"/api/v1/uploads/{upload}/content", content=b"abcd").status_code == 200
    assert client.post(f"/api/v1/uploads/{upload}/complete").status_code == 409
    quoted = client.post(f"/api/v1/conversations/{draft['conversation_id']}/inspect").json()
    assert quoted["quote"]["amount_cents"] == 30
    with container.store.transaction() as tx:
        assert tx.query(Job) == [] and tx.query(LedgerEntry) == []
    assert confirm(client, quoted).status_code == 402
    fund(container)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: confirm(client, quoted), range(2)))
    assert all(r.status_code == 200 for r in results)
    assert results[0].json()["job_id"] == results[1].json()["job_id"]
    assert client.get("/api/v1/me").json()["balance_cents"] == 970
    with container.store.transaction() as tx:
        assert len(tx.query(Job)) == len(tx.query(LedgerEntry)) == 1


def test_ownership_and_verification(client):
    draft = create(client)
    path = f"/api/v1/conversations/{draft['conversation_id']}"
    assert client.get(path, headers={"Authorization":"Bearer fake:other"}).status_code == 403
    assert client.post(path + "/confirm", json={"revision":0}, headers={"Authorization":"Bearer fake:other"}).status_code == 403
    assert client.post("/api/v1/conversations", json={}, headers={"Authorization":"Bearer fake:x:unverified"}).status_code == 403
    assert TestClient(client.app).get(path).status_code == 401


@pytest.mark.parametrize('status', ['succeeded', 'failed', 'cancelled', 'expired'])
def test_continue_keeps_conversation_and_previous_task_but_resets_quote(client, container, status):
    draft = edit(client, create(client), filename='first.mp4', size_bytes=4)
    draft = edit(client, draft, choice='locale', value='zh')
    key = draft['conversation_id']
    with container.store.transaction() as tx:
        current = tx.get(Conversation, key)
        current.status, current.job_id = 'submitted', 'previous-job'
        current.duration_ms, current.quoted_cents, current.upload_id = 90000, 15, 'old-upload'
        tx.put(current, key)
        tx.insert(Job(job_id='previous-job',owner_user_id='chat-user',status=status,original_filename='first.mp4'), 'previous-job')
    before_balance = client.get('/api/v1/me').json()['balance_cents']
    response = client.post(f'/api/v1/conversations/{key}/continue',json={'revision':draft['revision']})
    assert response.status_code == 200, response.text
    next_round = response.json()
    assert next_round['conversation_id'] == key and next_round['step'] == 'source'
    assert next_round['locale'] == next_round['explicit_locale'] == 'zh'
    assert next_round['messages'][:-1] == draft['messages']
    assert next_round['messages'][-1]['job_id'] == 'previous-job'
    assert not any(next_round[field] for field in ['job_id','upload_id','quoted_cents','duration_ms','target_language','filename'])
    assert 'quote' not in next_round and 'job' not in next_round
    assert confirm(client, next_round).status_code == 409
    assert client.post(f'/api/v1/conversations/{key}/continue',json={'revision':draft['revision']}).status_code == 409
    next_round = edit(client, next_round, filename='second.mp3',size_bytes=8)
    assert next_round['step'] == 'target'
    assert client.get('/api/v1/me').json()['balance_cents'] == before_balance
    with container.store.transaction() as tx:
        assert len(tx.query(Job)) == 1 and not tx.query(LedgerEntry)


def test_continue_rejects_active_tasks_and_other_users(client, container):
    draft = create(client)
    key = draft['conversation_id']
    with container.store.transaction() as tx:
        current = tx.get(Conversation, key)
        current.status, current.job_id = 'submitted', 'active-job'
        tx.put(current, key)
        tx.insert(Job(job_id='active-job', owner_user_id='chat-user', status='running'),'active-job')
    path = f'/api/v1/conversations/{key}/continue'
    assert client.post(path,json={'revision':draft['revision']}).status_code == 409
    assert client.post(path,json={'revision':draft['revision']},headers={'Authorization':'Bearer fake:other'}).status_code == 403


@pytest.mark.parametrize("url", ["https://youtube.com.evil.test/watch?v=KlvBiyWSI-Y", "file:///etc/passwd", "http://localhost/a", "https://user@youtube.com/watch?v=KlvBiyWSI-Y", "https://youtube.com/playlist?list=abc", "https://youtu.be/no"])
def test_youtube_allowlist(url):
    with pytest.raises(ValueError):
        youtube_url(url)


def test_youtube_import_reuses_reservation_after_storage_failure(client, container):
    class Importer:
        calls = 0
        def download(self, url, destination, max_bytes):
            self.calls += 1
            assert url == URL
            video = destination / "video.mp4"
            video.write_bytes(b"fake video")
            return video
    importer = Importer()
    container.conversations.importer = importer
    draft = edit(client, create(client), text=f"翻译成中文 {URL}")
    confirmed = prepare(client, draft).json()
    upload = container.storage.upload
    container.storage.upload = lambda *_: (_ for _ in ()).throw(OSError("storage unavailable"))
    container.conversations.import_one()
    failed = client.get(f"/api/v1/conversations/{draft['conversation_id']}").json()
    assert failed["status"] == "import_failed"
    container.storage.upload = upload
    prepare(client, confirmed)
    container.conversations.import_one()
    done = client.get(f"/api/v1/conversations/{draft['conversation_id']}").json()
    assert done["status"] == "draft" and done["quote"]["amount_cents"] == 30
    with container.store.transaction() as tx:
        assert tx.query(Job) == []
    fund(container)
    assert confirm(client, done).json()["status"] == "submitted"
    assert done["upload_id"] == failed["upload_id"]
    container.conversations.import_one()
    assert importer.calls == 2
    with container.store.transaction() as tx:
        assert len(tx.query(Job)) == len(tx.query(UploadSession)) == 1


def test_sqlite_draft_and_confirmation_survive_restart(container, tmp_path):
    settings = replace(container.settings, store_path=str(tmp_path / "chat.db"))
    first = build_container(settings)
    service = first.conversations
    draft = service.create("u", "zh")
    draft = service.edit(draft.conversation_id, "u", 0, filename="video.mp4", size_bytes=5)
    draft = service.edit(draft.conversation_id, "u", 1, choice="target", value="en")
    confirmed = service.prepare(draft.conversation_id, "u", draft.revision)
    second = build_container(settings)
    restored = second.conversations.prepare(draft.conversation_id, "u", draft.revision)
    assert restored.upload_id == confirmed.upload_id
    assert restored.locale == "zh"


def test_model_cannot_invent_a_download_or_skip_review(client, container):
    class Interpreter:
        def interpret(self, text, context):
            return Interpretation(target_language="zh", reply="Ready", mode="ai")
    container.conversations.interpreter = Interpreter()
    draft = edit(client, create(client), text="start everything")
    assert draft["status"] == "draft" and not draft["ready"]


def test_unsupported_target_change_invalidates_previous_review(client, container):
    draft = edit(client, create(client), text=f"翻译成中文 {URL}")
    class Interpreter:
        def interpret(self, text, context):
            return Interpretation(target_language="unsupported", reply="Please choose Chinese or English", mode="ai")
    container.conversations.interpreter = Interpreter()
    draft = edit(client, draft, text="Actually make it French")
    assert not draft["ready"] and draft["step"] == "target"
    assert confirm(client, draft).status_code == 409


def test_deepseek_failure_falls_back_without_losing_understood_slots(monkeypatch):
    import httpx
    def fail(*args, **kwargs):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(httpx, "post", fail)
    interpreter = DeepSeekConversationInterpreter("secret", "model", "https://example.test")
    result = interpreter.interpret(f"翻译成中文 {URL}", {})
    assert result.mode == "fallback" and result.target_language == "zh" and result.youtube_url == URL


def test_static_entry_point(client):
    assert client.get("/").status_code == 200
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/api/v1/chat/config").json()["target_languages"] == ["zh", "en"]


def prepared_audio(client):
    draft = edit(client, create(client), filename="speech.mp3", size_bytes=4)
    draft = edit(client, draft, choice="target", value="en")
    draft = prepare(client, draft).json()
    assert client.put(f"/api/v1/uploads/{draft['upload_id']}/content", content=b"abcd").status_code == 200
    return client.post(f"/api/v1/conversations/{draft['conversation_id']}/inspect").json()


def test_audio_quote_frozen_media_and_final_confirmation(client, container):
    draft = prepared_audio(client)
    assert draft["media_type"] == "audio" and draft["quote"]["amount_cents"] == 30
    upload = container.jobs.get_upload(draft["upload_id"], "chat-user")
    assert "prepared_" in upload.object_key and "upload" not in draft
    assert client.put(f"/api/v1/uploads/{draft['upload_id']}/content", content=b"evil").status_code == 409
    assert client.post(f"/api/v1/uploads/{draft['upload_id']}/renew").status_code == 409
    fund(container)
    confirmed = confirm(client, draft).json()
    assert confirmed["job"]["media_type"] == "audio"
    with container.store.transaction() as tx:
        assert tx.query(JobOutbox)[0].jobspec["output_uri"].endswith(".mp3")
    assert client.get("/api/v1/me").json()["balance_cents"] == 970


def test_replacing_source_clears_quote_and_stale_confirmation(client, container):
    draft = prepared_audio(client)
    changed = edit(client, draft, choice="edit_source")
    assert changed["target_language"] == "en" and "quote" not in changed
    assert not changed["upload_id"] and not changed["job_id"]
    assert confirm(client, draft).status_code == 409
    assert confirm(client, changed).status_code == 409
    with container.store.transaction() as tx:
        assert tx.query(Job) == []


def test_old_quote_stays_locked_even_if_price_changes_during_confirmation(client, container):
    draft = prepared_audio(client)
    assert draft["quote"]["amount_cents"] == 30
    fund(container)
    original = container.funding.complete_inspection
    def change_then_charge(command):
        container.funding.pricing.update(expected_version=container.funding.pricing.current().pricing_version,
            rate=40, minimum=10, increment=10, actor="admin")
        return original(command)
    container.funding.complete_inspection = change_then_charge
    response = confirm(client, draft)
    assert response.status_code == 200
    assert response.json()["job"]["quoted_point_units"] == 30
    assert client.get("/api/v1/me").json()["balance_cents"] == 970
    assert response.json()["quote"]["rate_cents_per_minute"] == 20
    assert prepared_audio(client)["quote"]["amount_cents"] == 60


def test_target_edit_preserves_prepared_quote_and_uses_new_language(client, container):
    draft = prepared_audio(client)
    changed = edit(client, draft, choice="target", value="zh")
    assert changed["quote"] == draft["quote"]
    fund(container)
    assert confirm(client, changed).json()["job"]["target_language"] == "zh"


def test_quote_survives_sqlite_restart_and_confirmation_is_once(container, tmp_path):
    settings = replace(container.settings, store_path=str(tmp_path / "prepared.db"), pricing=Settings().pricing)
    first = build_container(settings)
    first.conversations.inspector = type("Probe", (), {"inspect":lambda self, path: MediaInspectionResult(90000,"90", "audio", True)})()
    http = TestClient(create_app(first), headers=AUTH)
    draft = prepared_audio(http)
    fund(first)
    second = build_container(settings)
    http2 = TestClient(create_app(second), headers=AUTH)
    assert confirm(http2, draft).status_code == 200
    assert confirm(http2, draft).status_code == 200
    assert http2.get("/api/v1/me").json()["balance_cents"] == 970


def test_crashed_inspection_is_recovered_without_charge(client, container):
    draft = prepared_audio(client)
    with container.store.transaction() as tx:
        row = tx.get(Conversation, draft["conversation_id"])
        row.duration_ms = row.quoted_cents = 0
        row.status, row.lease_until = "inspecting", 1
        tx.put(row, row.conversation_id)
    container.conversations.import_one()
    restored = client.get(f"/api/v1/conversations/{draft['conversation_id']}").json()
    assert restored["status"] == "draft" and restored["quote"]["amount_cents"] == 30
    with container.store.transaction() as tx:
        assert tx.query(Job) == [] and tx.query(LedgerEntry) == []
