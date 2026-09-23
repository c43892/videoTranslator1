# Conversational Studio

The v2 control API now serves the conversation UI at `/`. Users may choose an
input or describe the whole request. A persisted draft collects a single YouTube
video or local video/audio, a dubbing target, and a separate conversation language.
An explicit preparation step uploads/downloads and probes the media without
charging. The review shows a trusted quote; only final confirmation charges and
queues translation. Text such as “start now” cannot dispatch a job. Source edits
invalidate the quote; repeated confirmations share one job and one ledger debit.

## Run

```powershell
python -m pip install -e "packages/videotranslator[dev]"
New-Item -ItemType Directory -Force vt-data | Out-Null
$env:APP_PROFILE = 'local-ui'
$env:STORE_PATH = 'vt-data/studio.db'
$env:PORT = '8090'
python -m videotranslator.api
```

Set `DEEPSEEK_API_KEY` in the server environment to enable semantic input parsing.
`CHAT_MODEL` defaults to `deepseek-chat`, and `DEEPSEEK_BASE_URL` defaults to
`https://api.deepseek.com`. Keys never enter browser configuration or responses.
Without a key, or during a provider outage, guided choices and conservative
Chinese/English parsing remain available. Provider errors never start jobs.
FFmpeg and yt-dlp are needed for YouTube imports; yt-dlp is a package dependency.
Some YouTube videos may require a compatible JavaScript runtime or be unavailable
to the deployment's IP. Failed downloads remain visible and can be retried.

Container preview:

```powershell
# Fill .env from .env.example if needed.
docker compose -f compose.chat.yml up --build
```

The preview binds only to localhost:8090 and persists drafts in a named volume.
The control/UI image is separate from the GPU worker image. `local-ui` rejects
translation confirmation before any task debit. The real local deployment uses
`local-full` with `ENGINE_BACKEND=docker` to connect to the containerized engine
from `codex/fresh-start`; see [LOCAL_ENGINE.md](LOCAL_ENGINE.md) for startup,
contracts and verification. Only the automated `test` profile simulates success.

On local API startup, legacy successful jobs are corrected only when both their
backend ID identifies the fake backend and the result is exactly the 11-byte
`fake-result` placeholder. The task becomes failed, its download is disabled,
and the original debit returns to account balance exactly once. Input media,
task history, payment records and placeholder files are retained.

## Languages

Initial guidance uses the browser's preferred language (which normally inherits
the system setting). Built-in UI catalogs: Chinese, English, French, Spanish,
German, Japanese, Korean, Portuguese; other locales fall back to English.
The first and subsequent textual replies can change the conversation language.
An explicit “reply in …” request or language menu selection takes precedence and
persists for that conversation until Auto is selected. Bare links, filenames,
uploads and target-language buttons do not trigger language detection.
Video dubbing targets remain Chinese and English, independently of UI language.

## Boundaries and integration

- `ConversationInterpreter` extracts intent without tools or execution authority.
  DeepSeek is wired only in `bootstrap.py`; it can be replaced independently.
- `VideoSourceImporter` supplies a local media file. The yt-dlp implementation
  accepts canonical single-video YouTube URLs only, limits size and duration,
  and runs in a timed subprocess. No user-supplied cookies or shell arguments.
- `ConversationService` owns draft revisions and durable import claims. Background
  imports use a 20-minute lease and a 15-minute subprocess timeout. An interrupted
  import becomes eligible after its lease expires; retry reuses its reservation.
  Imports join the existing upload inspection/funding/job queue, never call GPUs.
- Local upload transport streams to an owned staging file with exact byte limits.
  Azure uses the existing signed upload URL directly. Interrupted uploads can
  restart using the same reservation; page refresh requires reselecting the same
  local file. This version restarts the transfer, not a byte-range resume.
- Firebase Authentication is built into the UI (Google and email/password,
  registration, verification, password reset, logout). The backend verifies
  signed ID tokens, revocation, and verified email before media or payment work.
  `AUTH_MODE=firebase` is the default, even on localhost. Explicit local-only
  `AUTH_MODE=demo` permits simulated identities and disables payment gateways.
- Account balance is always in the header and refreshed from the server.
  Private conversations are stored under the Firebase UID, not a shared browser
  draft key. The public config exposes only Firebase web configuration.
- Stripe/PayPal top-ups and the quote/review flow are documented in
  [ACCOUNTS_AND_BILLING.md](ACCOUNTS_AND_BILLING.md).
- This repository uses SQLite/MemoryStore. Multi-replica cloud operation requires
  the planned shared transactional store; copying separate SQLite files to each
  replica is not a shared queue.

## Checks

`python -m pytest tests/unit` covers draft isolation, no work before confirmation,
stale review rejection, concurrent confirmation, language precedence, safe URL
validation, upload byte limits, import retries and SQLite restart persistence.
Browser checks exercise natural-language input, review editing, reload recovery,
and uploading a synthetic one-second video through the actual UI.

API: `POST /api/v1/conversations`, `GET /{id}`, `POST /{id}/messages`,
`POST /{id}/prepare`, `POST /{id}/inspect`, `POST /{id}/confirm`; all paths after the first are relative to
`/api/v1/conversations`. Existing upload and job APIs remain available to other
clients. Conversational reservations cannot use the legacy upload-complete
endpoint to bypass the final quote confirmation.
# Conversation continuation and task sidebar

The login dialog follows the active interface language and includes its own flag
language picker (Auto, Chinese, English). Both the interface and assistant replies
support only Chinese and English. Other browser languages and previously saved
unsupported language preferences fall back to English. It shares the persisted preference with the header picker, without
resetting form values. Firebase `auth.languageCode` follows that language (Chinese
maps to `zh-CN`) for its OAuth flow and account emails. The Google sign-in button
remains visible alongside email sign-in. Each picker has a distinct accessible
listbox ID. Login errors and notices update when the language changes.

Firebase verification distinguishes invalid/revoked credentials (401) from
temporary verification failures (503 `auth_unavailable`). Cold initialization is
serialized; transient verification failures get one retry with signature and
revocation checks retained. Logs record exception types, never tokens or SDK
error payloads. The browser refreshes credentials once on an explicit 401 and
retries explicit pre-handler auth outages twice. General 5xx/transport failures
are not replayed, since a mutation could already have committed. Account changes
invalidate requests in flight. Failed conversation restoration remains retryable
without signing out, and balance loading is independent of that restoration.

After a completed, failed, cancelled, or expired task, the composer accepts a new
link or attachment. Users can also choose “Translate another file.” The
revision-checked `/conversations/{id}/continue` endpoint retains the conversation,
language preference, messages, and a link to the previous task, while clearing the
source, upload reservation, target language and quote. Each new round requires its
own inspection, price review and confirmation; continuation never charges.
Interpretation uses only messages from the current input round.

The sidebar provides Conversation and Task history navigation. Task history
always opens the complete list in the main content area, clearing any previously
selected detail. Rows expose status, language, duration, cost, date and downloads.
Details occupy the main content area with a Back to task list control that retains
the filter and list scroll position. Returning to Conversation preserves the
current draft and unsent composer text; only New conversation creates a new draft.

The sidebar contains only Conversation and Task history; task rows appear once,
in the main history view. On narrow screens navigation becomes a compact
two-button bar and task rows become cards. Status polling continues
every five seconds while the page is visible. Task details support native result
playback; leaving details stops playback. Balance remains visible in every view.

The single header combines balance, top-up, language and an account menu. Refresh
and sign-out are in that menu. The entry screen presents a short prompt, the rate
and source choices; after the first substantive message the introduction hides.
New conversation appears only once there is a conversation to replace. Guests
see a single sign-in action rather than a disabled composer. The auth dialog keeps
Google/email sign-in and its language picker, with troubleshooting help collapsed.
Review retains duration, rate, total charge, remaining balance and explicit start
confirmation. The history list omits internal IDs; they remain in task details.


### Recovering task-status notices (2026-09-23)

A brief Firebase rejection could leave a permanent-looking sign-in warning even
after `/me`, task history and conversation polling had returned to HTTP 200.
The shared error notice now clears authentication warnings after a successful
authenticated response and clears transient background warnings when their own
poll recovers. Unrelated user-action errors are preserved. While a task is active,
a short outage is described as reconnecting status updates; persistent credential
rejections still request sign-in and explain that the backend task continues.
Credential verification, signature checks and revocation checks are unchanged.
Rejection logs include only a fixed reason category, never token or SDK contents.
