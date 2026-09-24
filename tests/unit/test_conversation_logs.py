from contextlib import contextmanager
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from videotranslator.api.app import create_app
from videotranslator.bootstrap import build_container
from videotranslator.docstore import SQLiteStore
from videotranslator.domain.conversation import Conversation, ConversationLog
from videotranslator.domain.models import Job, LedgerEntry
from .test_conversations import AUTH, URL, client, create, edit


def logs(container, key):
    with container.store.transaction() as tx:
        return tx.query(ConversationLog, where=("conversation_id", "==", key), order_by="revision")


def test_all_turns_archived_once_without_charges_or_feedback_api(client, container):
    draft = create(client, "zh")
    for text in [f"翻译成英文 {URL}", "这个功能不好用", "陪我聊聊天", "改成法语"]:
        old = draft
        draft = edit(client, draft, text=text)
        # A repeated request using the old revision must not insert a second log.
        replay = client.post(f"/api/v1/conversations/{draft['conversation_id']}/messages",
                             json={"revision": old["revision"], "text": text})
        assert replay.status_code == 409
    entries = logs(container, draft["conversation_id"])
    assert [item.event for item in entries] == ["created"] + ["message"] * 4
    assert [item.intent for item in entries[1:]] == ["product", "complaint", "off_topic", "product"]
    for index, entry in enumerate(entries[1:]):
        assert entry.owner_user_id == "chat-user" and entry.created_at > 0
        assert entry.messages == draft["messages"][index * 2:index * 2 + 2]
        assert "c43892@gmail.com" not in entry.messages[-1]["text"]
        assert "意见反馈" not in entry.messages[-1]["text"]
    assert client.get("/api/v1/feedback").status_code == 404
    assert client.get("/api/v1/conversation-logs").status_code == 404
    with container.store.transaction() as tx:
        assert not tx.query(Job) and not tx.query(LedgerEntry)


def test_new_conversation_and_locale_change_preserve_old_logs(client, container):
    first = edit(client, create(client), text="The dubbing sounds unnatural")
    changed = edit(client, first, choice="locale", value="zh")
    second = create(client)
    entries = logs(container, first["conversation_id"])
    assert len(entries) == 3 and entries[-1].event == "locale"
    assert entries[-1].messages[0]["value"] == "zh"
    assert entries[1].messages == first["messages"] == changed["messages"]
    assert len(logs(container, second["conversation_id"])) == 1
    denied = client.post(f"/api/v1/conversations/{first['conversation_id']}/messages",
        headers={"Authorization": "Bearer fake:another-user"},
        json={"revision": changed["revision"], "text": "这个功能不好用"})
    assert denied.status_code == 403
    assert len(logs(container, first["conversation_id"])) == 3


def test_archive_failure_rolls_back_conversation(client, container, monkeypatch):
    draft = create(client)
    transaction = container.store.transaction
    @contextmanager
    def failing_transaction():
        with transaction() as tx:
            insert = tx.insert
            def fail_log(doc, key):
                if isinstance(doc, ConversationLog):
                    raise RuntimeError("archive unavailable")
                return insert(doc, key)
            tx.insert = fail_log
            yield tx
    monkeypatch.setattr(container.store, "transaction", failing_transaction)
    with pytest.raises(RuntimeError, match="archive unavailable"):
        container.conversations.edit(draft["conversation_id"], "chat-user", draft["revision"], text="这个功能不好用")
    with transaction() as tx:
        saved = tx.get(Conversation, draft["conversation_id"])
        assert saved.revision == draft["revision"] and not saved.messages
        assert len(tx.query(ConversationLog)) == 1


def test_sqlite_archive_survives_service_restart(container, tmp_path):
    settings = replace(container.settings, store_path=str(tmp_path / "logs.db"))
    first = build_container(settings)
    api = TestClient(create_app(first), headers=AUTH)
    draft = edit(api, create(api), text="The translation is inaccurate")
    first.store._conn.close()
    reopened = SQLiteStore(settings.store_path)
    with reopened.transaction() as tx:
        entries = tx.query(ConversationLog, where=("conversation_id", "==", draft["conversation_id"]))
        assert len(entries) == 2
        message = next(entry for entry in entries if entry.event == "message")
        assert message.messages == draft["messages"] and message.intent == "complaint"
    reopened._conn.close()
