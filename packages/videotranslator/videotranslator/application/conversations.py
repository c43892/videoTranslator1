"""Conversational draft state, explicit confirmation, and durable import work."""
import re
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from ..conversation_ports import ConversationInterpreter, Interpretation, VideoSourceImporter
from ..domain.conversation import Conversation
from ..domain.enums import Forbidden, InsufficientCredits, InvalidTransition, NotFound, StaleVersion
from ..domain.models import Job, UploadSession, User, now_ms
from ..domain.pricing import quote_job
from .uow import CompleteInspectionCommand
from ..ids import new_id


def locale(value: str) -> str:
    primary = value.lower().split("-")[0] if re.fullmatch(r"[a-zA-Z]{2,3}(?:-[a-zA-Z0-9]{2,8})*", value) else "en"
    return primary if primary in {"zh", "en"} else "en"


class ConversationService:
    def __init__(self, store, jobs, storage, settings, interpreter: ConversationInterpreter,
                 importer: VideoSourceImporter, validate_url, inspector, funding):
        self.store, self.jobs, self.storage, self.settings = store, jobs, storage, settings
        self.interpreter, self.importer, self.validate_url = interpreter, importer, validate_url
        self.inspector, self.funding = inspector, funding

    def create(self, owner: str, browser_locale: str) -> Conversation:
        draft = Conversation(conversation_id=new_id("chat"), owner_user_id=owner, locale=locale(browser_locale),
                             interpreter_mode=getattr(self.interpreter, "mode", "guided"))
        with self.store.transaction() as tx:
            tx.insert(draft, draft.conversation_id)
        return self.get(draft.conversation_id, owner)

    def owned(self, tx, key, owner):
        draft = tx.get(Conversation, key)
        if draft is None:
            raise NotFound("conversation_not_found")
        if draft.owner_user_id != owner:
            raise Forbidden()
        draft.locale = locale(draft.locale)
        if draft.explicit_locale:
            draft.explicit_locale = locale(draft.explicit_locale)
        return draft

    def get(self, key, owner):
        with self.store.transaction() as tx:
            return self.owned(tx, key, owner)

    def edit(self, key, owner, revision, *, text="", choice="", value="", filename="", size_bytes=0):
        before = self.get(key, owner)
        language_only = choice == "locale" and not text and not filename
        if before.status != "draft" and not language_only:
            raise InvalidTransition("already_confirmed")
        if before.revision != revision:
            raise StaleVersion("draft_changed")
        if len(before.messages) - before.round_start >= 100:
            raise ValueError("conversation_limit")
        parsed = self.interpreter.interpret(text, {
            "locale": before.explicit_locale or before.locale, "source_kind": before.source_kind,
            "target_language": before.target_language, "messages": before.messages[before.round_start:][-6:],
        }) if text else Interpretation()
        with self.store.transaction() as tx:
            draft = self.owned(tx, key, owner)
            if draft.revision != revision or (draft.status != "draft" and not language_only):
                raise StaleVersion("draft_changed")
            if parsed.explicit_locale:
                draft.explicit_locale = locale(parsed.explicit_locale)
            if parsed.detected_locale and not draft.explicit_locale:
                draft.locale = locale(parsed.detected_locale)
            if draft.explicit_locale:
                draft.locale = draft.explicit_locale
            source = value if choice == "source" else parsed.source_kind
            if source in {"upload", "youtube"} and source != draft.source_kind:
                draft.source_kind, draft.youtube_url, draft.filename, draft.size_bytes = source, "", "", 0
            if parsed.youtube_url:
                draft.youtube_url = self.validate_url(parsed.youtube_url)
                draft.source_kind, draft.filename, draft.size_bytes = "youtube", "", 0
            target = value if choice == "target" else parsed.target_language
            if choice != "target" and target in {"unsupported", "unclear"}:
                draft.target_language, target = "", ""
            if target:
                if target not in {"zh", "en"}:
                    raise ValueError("unsupported_target")
                draft.target_language = target
            if choice == "locale":
                draft.explicit_locale = "" if value == "auto" else locale(value)
                if draft.explicit_locale:
                    draft.locale = draft.explicit_locale
            if choice == "edit_source":
                draft.source_kind, draft.youtube_url, draft.filename, draft.size_bytes = "", "", "", 0
            if choice == "edit_target":
                draft.target_language = ""
            if filename:
                if Path(filename).suffix.lower() not in {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"}:
                    raise ValueError("video_required")
                if not 0 < size_bytes <= self.settings.max_upload_bytes:
                    raise ValueError("invalid_file_size")
                draft.source_kind, draft.filename, draft.size_bytes = "upload", Path(filename).name, size_bytes
                draft.youtube_url = ""
            if filename or (draft.source_kind, draft.youtube_url) != (before.source_kind, before.youtube_url):
                draft.upload_id = draft.job_id = draft.pricing_version = ""
                draft.duration_ms = draft.quoted_cents = 0
            draft.revision += 1
            draft.interpreter_mode = parsed.mode if text else draft.interpreter_mode
            if choice != "locale":
                draft.messages.append({"role": "user", "text": text, "choice": choice, "value": value, "filename": filename})
                # Assistant prose is display-only; it cannot affect readiness or confirmation.
                draft.messages.append({"role": "assistant", "text": parsed.reply[:800], "step": self.step(draft)})
            tx.put(draft, key)
        return self.get(key, owner)

    @staticmethod
    def step(draft):
        if draft.status != "draft":
            return draft.status
        if not draft.source_kind:
            return "source"
        if draft.source_kind == "youtube" and not draft.youtube_url:
            return "link"
        if draft.source_kind == "upload" and not draft.filename:
            return "file"
        return "review" if draft.target_language else "target"

    def prepare(self, key, owner, revision):
        with self.store.transaction() as tx:
            draft = self.owned(tx, key, owner)
            if draft.revision != revision:
                raise StaleVersion("draft_changed")
            if not draft.ready:
                raise InvalidTransition("draft_incomplete")
            if draft.status == "draft" and not draft.duration_ms:
                if draft.source_kind == "upload":
                    upload = self.jobs.prepare_upload(owner, filename=draft.filename,
                        size_bytes=draft.size_bytes, target_language=draft.target_language)
                    tx.insert(upload, upload.upload_id)
                    draft.upload_id, draft.job_id = upload.upload_id, upload.reserved_job_id
                    draft.status = "uploading"
                else:
                    draft.status = "import_pending"
                tx.put(draft, key)
            elif draft.status == "import_failed":
                draft.status, draft.error = "import_pending", ""
                tx.put(draft, key)
        return self.get(key, owner)

    def finish_preparation(self, key, owner):
        with self.store.transaction() as tx:
            draft = self.owned(tx, key, owner)
            if draft.duration_ms:
                return draft
            expired_inspection = draft.status == "inspecting" and draft.lease_until <= now_ms()
            if draft.status not in {"uploading", "importing"} and not expired_inspection:
                raise InvalidTransition("media_not_ready")
            draft.status = "inspecting"
            draft.lease_until = now_ms() + 20 * 60 * 1000
            tx.put(draft, key)
        session = self.jobs.get_upload(draft.upload_id, owner)
        try:
            if not self.storage.exists(session.object_key, expected_size=session.declared_size_bytes):
                raise ValueError("upload_incomplete")
            # Freeze the exact bytes quoted. Outstanding upload URLs cannot mutate this object.
            with TemporaryDirectory(prefix="vt-quote-") as directory:
                media = self.storage.download(session.object_key, Path(directory) / session.original_filename)
                result = self.inspector.inspect(media)
                quote = quote_job(self.settings.pricing, result.duration_ms, result.media_type)
                frozen_key = f"users/{owner}/jobs/{session.reserved_job_id}/{new_id('prepared')}{Path(session.original_filename).suffix}"
                self.storage.upload(media, frozen_key)
            with self.store.transaction() as tx:
                current = self.owned(tx, key, owner)
                if current.status != "inspecting" or current.lease_until != draft.lease_until:
                    raise StaleVersion("draft_changed")
                upload = tx.get(UploadSession, session.upload_id)
                upload.object_key, upload.status = frozen_key, UploadSession.STATUS_COMMITTED
                tx.put(upload, upload.upload_id)
                current.duration_ms, current.duration_probe_raw = result.duration_ms, result.duration_probe_raw
                current.media_type = str(result.media_type)
                current.quoted_cents, current.pricing_version = quote.point_units, quote.pricing_version
                current.status, current.lease_until, current.error = "draft", 0, ""
                current.revision += 1
                tx.put(current, key)
        except Exception:
            with self.store.transaction() as tx:
                current = self.owned(tx, key, owner)
                if current.lease_until == draft.lease_until and current.status == "inspecting":
                    current.status = "uploading" if current.source_kind == "upload" else "import_failed"
                    current.lease_until = 0
                    tx.put(current, key)
            raise
        return self.get(key, owner)

    def confirm(self, key, owner, revision):
        with self.store.transaction() as tx:
            draft = self.owned(tx, key, owner)
            if draft.revision != revision:
                raise StaleVersion("draft_changed")
            if draft.status == "submitted":
                return draft
            self.funding.require_processing()
            if not draft.ready or not draft.duration_ms or not draft.upload_id:
                raise InvalidTransition("quote_required")
            quote = quote_job(self.settings.pricing, draft.duration_ms, draft.media_type)
            if (quote.pricing_version, quote.point_units) != (draft.pricing_version, draft.quoted_cents):
                raise StaleVersion("price_changed")
            if draft.status not in {"draft", "confirming"}:
                raise InvalidTransition("media_not_ready")
            user = tx.get(User, owner)
            existing = tx.get(Job, draft.job_id)
            if not existing and (not user or user.point_balance_units < draft.quoted_cents):
                raise InsufficientCredits()
            draft.status = "confirming"
            tx.put(draft, key)
        session = self.jobs.get_upload(draft.upload_id, owner)
        self.funding.complete_inspection(CompleteInspectionCommand(
            job_id=draft.job_id, owner_user_id=owner, original_filename=session.original_filename,
            media_type=draft.media_type, target_language=draft.target_language,
            input_object_key=session.object_key, duration_ms=draft.duration_ms,
            duration_probe_raw=draft.duration_probe_raw, inspection_attempt=1))
        with self.store.transaction() as tx:
            current = self.owned(tx, key, owner)
            current.status = "submitted"
            tx.put(current, key)
            upload = tx.get(UploadSession, draft.upload_id)
            upload.status, upload.completed_at = UploadSession.STATUS_COMPLETED, now_ms()
            tx.put(upload, upload.upload_id)
        return self.get(key, owner)

    def view(self, draft):
        result = asdict(draft)
        result.pop("owner_user_id")
        result.pop("lease_until")
        result.update(ready=draft.ready, step=self.step(draft))
        if draft.upload_id:
            session = self.jobs.get_upload(draft.upload_id, draft.owner_user_id)
            if session.status == UploadSession.STATUS_UPLOADING:
                result["upload"] = asdict(self.jobs.upload_ticket(session))
        if draft.duration_ms:
            result["quote"] = {"duration_ms": draft.duration_ms, "amount_cents": draft.quoted_cents,
                "rate_cents_per_minute": self.settings.pricing.point_units_per_minute,
                "currency": "USD", "pricing_version": draft.pricing_version}
        if draft.job_id:
            with self.store.transaction() as tx:
                job = tx.get(Job, draft.job_id)
            if job:
                result["job"] = asdict(job)
        return result

    def follow_retry(self, key, owner, job_id):
        job = self.jobs.get_job(job_id, owner)
        with self.store.transaction() as tx:
            draft = self.owned(tx, key, owner)
            if draft.status != 'submitted' or (job.job_id != draft.job_id and job.retry_of_job_id != draft.job_id):
                raise InvalidTransition("invalid_retry")
            if draft.job_id != job.job_id:
                draft.revision += 1
            draft.job_id = job.job_id
            tx.put(draft, key)
        return self.get(key, owner)

    def continue_conversation(self, key, owner, revision):
        """Start a fresh input round without deleting the conversation or charging."""
        with self.store.transaction() as tx:
            previous = self.owned(tx, key, owner)
            if previous.revision != revision:
                raise StaleVersion("draft_changed")
            job = tx.get(Job, previous.job_id) if previous.job_id else None
            if (previous.status != "submitted" or not job or job.owner_user_id != owner
                    or job.status not in {"succeeded", "failed", "cancelled", "expired"}):
                raise InvalidTransition("already_confirmed")
            messages = [*previous.messages, {"role": "assistant", "step": "previousTask",
                "job_id": job.job_id, "filename": job.original_filename, "status": str(job.status)}]
            draft = Conversation(conversation_id=key, owner_user_id=owner,
                revision=revision + 1, locale=previous.locale, explicit_locale=previous.explicit_locale,
                interpreter_mode=previous.interpreter_mode, messages=messages, round_start=len(messages))
            tx.put(draft, key)
        return self.get(key, owner)

    def import_one(self):
        """CPU source work. Claim a durable lease; no GPU calls or in-memory job queue."""
        now = now_ms()
        with self.store.transaction() as tx:
            interrupted = tx.query(Conversation, where=("status", "==", "inspecting"))
        for item in interrupted:
            if item.lease_until <= now:
                self.finish_preparation(item.conversation_id, item.owner_user_id)
                return
        with self.store.transaction() as tx:
            candidates = tx.query(Conversation, where_in=("status", ["import_pending", "importing"]))
            draft = next((d for d in candidates if d.lease_until <= now), None)
            if draft is None:
                return
            draft = tx.get(Conversation, draft.conversation_id)
            draft.status, draft.lease_until = "importing", now + 20 * 60 * 1000
            tx.put(draft, draft.conversation_id)
        key, owner = draft.conversation_id, draft.owner_user_id
        try:
            with TemporaryDirectory(prefix="vt-youtube-") as directory:
                # Retrying a completed upload only resolves its existing job.
                if not draft.upload_id:
                    video = self.importer.download(draft.youtube_url, Path(directory), self.settings.max_upload_bytes)
                    upload = self.jobs.prepare_upload(owner, filename=video.name,
                        size_bytes=video.stat().st_size, target_language=draft.target_language)
                    with self.store.transaction() as tx:
                        current = self.owned(tx, key, owner)
                        current.upload_id, current.job_id = upload.upload_id, upload.reserved_job_id
                        tx.insert(upload, upload.upload_id)
                        tx.put(current, key)
                    draft.upload_id = upload.upload_id
                    self.storage.upload(video, upload.object_key)
                else:
                    upload = self.jobs.get_upload(draft.upload_id, owner)
                    if not self.storage.exists(upload.object_key, expected_size=upload.declared_size_bytes):
                        video = self.importer.download(draft.youtube_url, Path(directory), self.settings.max_upload_bytes)
                        # A source can change between attempts; retain the reservation, update its size.
                        with self.store.transaction() as tx:
                            session = tx.get(UploadSession, upload.upload_id)
                            session.declared_size_bytes = video.stat().st_size
                            tx.put(session, session.upload_id)
                        self.storage.upload(video, upload.object_key)
                self.finish_preparation(key, owner)
        except Exception:
            with self.store.transaction() as tx:
                current = self.owned(tx, key, owner)
                current.status, current.error, current.lease_until = "import_failed", "youtube_download_failed", 0
                tx.put(current, key)
