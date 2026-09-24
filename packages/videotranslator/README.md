# VideoTranslator 2.0

The API now serves a conversational video-input studio at `/`. See
[`docs/CONVERSATIONAL_STUDIO.md`](../../docs/CONVERSATIONAL_STUDIO.md) for local and
container startup, language behavior, confirmation guarantees and integration limits.

Control plane + worker for long-running media translation, rebuilt per
`docs/PRODUCT_ARCHITECTURE_AND_IMPLEMENTATION.md` (v0.5).

## Layout

```
packages/videotranslator/videotranslator/
├─ domain/        # pure: models, state machine, pricing, DurationMatcher policy
├─ ports.py       # protocols: Store/Tx, identity, backends, storage, payments, processing
├─ docstore.py    # Store implementations: MemoryStore (test) + SQLiteStore (local)
├─ application/   # UoW (charge/cancel/refund/payment), services, dispatcher, reconciler
├─ adapters/      # fake · local · ffmpeg · stripe/paypal · azure (lazy SDK) · auth
├─ api/           # FastAPI control API (thin HTTP layer)
├─ worker/        # pipeline + GPU worker CLI
├─ bootstrap.py   # composition root: the only place adapters are chosen per profile
└─ inspection_worker.py  # CPU ffprobe worker CLI
tests/unit/       # §18.1 suite: 41 tests, no cloud, no GPU
```

Decoupling rules enforced by construction:

- `domain` imports nothing outside itself.
- Services depend on protocols (`ports.py`), never on concrete adapters.
- `bootstrap.py` is the **only** module that picks an adapter per profile
  (`test` / `local-ui` / `local-full` / `azure-*`); swap Whisper in, Stripe
  out — nothing else changes.
- All money/Point writes go through a Unit of Work = one store transaction.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/pip install -e packages/videotranslator
.venv/Scripts/python -m pytest tests/unit -q        # 41 tests
APP_PROFILE=local-full .venv/Scripts/python -m videotranslator.api   # :8000
```

The test profile uses `fake:<uid>` / `fake:<uid>:unverified`. Local and cloud
profiles default to verified Firebase ID tokens. Explicit local-only
`AUTH_MODE=demo` enables emulator identities and disables real payments.
See [account and billing setup](../../docs/ACCOUNTS_AND_BILLING.md) for
Google/email login, Stripe/PayPal top-ups, and the $0.20/minute audio/video flow.

## What's implemented vs. stubbed

- **Full:** state machine, Decimal pricing, charge/cancel/refund UoW with
  idempotency keys, dispatcher/cancel race handling, retry chains with cap,
  payment inbox (Stripe webhook + PayPal server capture), capacity counter,
  cost reservations, reconciler + deadline watchdog, local subprocess
  backends with SQLite recovery, FFmpeg inspection/renderer/duration-fit,
  worker pipeline with fake heavy models.
- **Profile adapters (cloud-only, lazily imported):** Azure Blob, Azure ML,
  Container Apps inspection, live Stripe/PayPal, Firebase Admin auth.
- **To slot in (§6.7):** real Whisper / Demucs / DeepSeek / IndexTTS2
  implementations of the processing ports — the pipeline factory in
  `worker/cli.py` is the single place to wire them.
