# Accounts and billing

## Product rules

- USD balance uses integer cents (`point_balance_units` remains the internal
  ledger field; `/me.balance_cents` is the public display amount).
- Video and audio cost $0.10 per minute of trusted server-probed duration.
  Round the final amount up to the next cent; minimum $0.01. For example,
  60 seconds costs $0.10, 90 seconds $0.15, and 61 seconds $0.11.
- Supported local media: MP4, MKV, MOV, WebM, AVI, M4V; MP3, WAV, M4A,
  FLAC, AAC, OGG. The existing 30-minute / 2 GiB limits still apply.
- Preparation uploads/downloads, probes, and freezes the exact quoted bytes.
  It creates no translation job and performs no balance debit. Review shows
  duration, rate, total charge, current balance and remaining balance.
- Final confirmation charges and queues one job. Duplicate requests cannot
  duplicate that debit. Insufficient balance returns to top-up, with the quote
  retained. Changing the source invalidates the quote; changing the language
  preserves it. Retrying a failed job shows its saved price before charging.
- Top-up packages (cash paid → account credit): $1 → $1; $10 → $11 (+10%);
  $50 → $60 (+20%); $100 → $130 (+30%). Bonus tiers use versioned v2 package IDs.
  Pending payments keep their saved amount and credit; payment verification checks
  cash paid while the ledger credits the saved balance including bonus exactly once.
  Top-ups are non-refundable; no cash refund
  endpoint exists and live gateway refund methods reject requests.
- Every failed task restores its actual ledger debit to usable account balance,
  exactly once, for the next translation. Uncharged tasks create no credit.
  Failure status, returned amount/time, balance and ledger are updated in one
  transaction. Reconciliation also repairs older failed tasks with unreturned
  debits. Successful tasks are never credited by late failure notifications.
- These are internal balance returns, never Stripe/PayPal refunds or payment
  reversals. Top-up payments remain successful. The existing cancellation policy
  is unchanged. Task details display the returned amount and time.

## Firebase setup

1. Create/select a Firebase project, register a Web app, enable Authentication
   providers **Google** and **Email/Password**, and configure authorized domains
   including localhost for development and the production host.
2. Fill `FIREBASE_API_KEY`, `FIREBASE_AUTH_DOMAIN`, `FIREBASE_PROJECT_ID`, and
   `FIREBASE_APP_ID` in `.env`. These are public Web app configuration values.
3. Place the server service-account JSON at `secrets/firebase-admin.json` (ignored
   by Git). Compose mounts `FIREBASE_ADMIN_CREDENTIALS_DIR` read-only at
   `/run/secrets`; native Python needs `GOOGLE_APPLICATION_CREDENTIALS` set to
   the actual host file path. Cloud workloads may instead use application
   default credentials with the appropriate Firebase permissions.
4. Keep `AUTH_MODE=firebase`. Missing configuration displays a clear unavailable
   notice; it does not silently create a demo user. Email accounts must verify
   their address before preparing media or purchasing balance.

Firebase's SDK manages the session and refreshing ID tokens. The API uses
Firebase Admin to check signatures and revocation. Firebase UID owns each
balance, payment, conversation and job. The application never stores passwords.

### Configured development project (2026-09-22)

- Firebase project: `videotranslator-64f6d` (VideoTranslator), Spark plan.
- Registered Web app: VideoTranslator Web. Google and Email/Password providers
  are enabled, with `localhost` authorized for development.
- Public SDK configuration is populated in the ignored local `.env`.
- The dedicated `videotranslator-auth` service account has only
  `roles/firebaseauth.viewer`; its ignored JSON is mounted by Compose and used
  by native Python through `GOOGLE_APPLICATION_CREDENTIALS`.
- A real Firebase Admin user-list request succeeded using that credential.
  The restarted port 8090 preview serves Firebase configuration and loads both
  sign-in options. Google sign-in subsequently completed in external Chrome:
  the application shows the authenticated account and its $0.00 balance, and
  authenticated `/me` and conversation requests succeeded.
- The Codex sidebar browser did not complete the Google popup flow. Use external
  Chrome/Edge for Google sign-in, or the email sign-in form. Sessions in separate
  browsers are independent. The login dialog explains this and provides an
  actionable message when popups are blocked or closed.
- Add the actual production hostname to authorized domains before deployment.
  No production hostname, payment activation, or billing upgrade was configured.

## Payments setup

Start with `PAYMENT_MODE=sandbox`, a persistent `STORE_PATH` and the correct
`PUBLIC_APP_URL` (browser return URL). Never share test balances with live users.
Use a separate database, Firebase project and payment credentials for testing.

### Stripe

Set `STRIPE_SECRET_KEY` (test key in sandbox) and `STRIPE_WEBHOOK_SECRET`.
Configure a webhook to `https://YOUR_HOST/api/v1/webhooks/stripe` for:

- `checkout.session.completed`
- `checkout.session.async_payment_succeeded`

For localhost, Stripe CLI can forward events to port 8090; its signing secret
must match the local configuration. Only signed, recent events with
`payment_status=paid`, matching currency and matching amount credit the account.

#### Configured local Stripe sandbox (2026-09-22)

- Merchant: **daybreak**, `acct_1UIYgDA9Wq7eEimk`.
- Its separate **daybreak sandbox**: `acct_1UIYgVAUTvR4hbMF`. The application's
  ignored `.env` uses only this sandbox's test key with `PAYMENT_MODE=sandbox`.
- Stripe CLI profile: `videotranslator-daybreak`, device `VideoTranslator-local`.
  CLI-issued credentials expire after 90 days and require reauthorization.
- The port 8090 native preview now uses
  `vt-data/chat-preview/store-stripe-sandbox.db`; the prior `store.db` is preserved.
  The authenticated user's $1.00 balance in this sandbox is test credit only.
- Run `.venv/Scripts/python.exe deploy/stripe-listen.py` to forward the two
  supported Checkout events locally. It saves the signing secret to `.env`
  without printing it; restart the API after a secret change. Keep one listener
  running while testing. This helper accepts only sandbox keys and localhost.
- Verified using hosted Stripe Checkout and the official test card: a $1 USD
  payment completed, the signed callback returned HTTP 200, and the ledger
  credited exactly 100 cents. Replaying the same event twice kept one purchase
  entry and the same 100-cent balance. No real funds were charged.
- The live activation draft reuses the existing business and payout bank details
  as requested; Stripe marks those form sections complete. This is not final
  activation or evidence of completed Stripe underwriting.
  The business category is Software as a service, with a video/audio translation
  description. The statement descriptor is DAYBREAK.
  The draft still contains the prior project's website and must be corrected
  with the user's actual public Daybreak website before submission.
  Final live activation, agreement submission and live keys remain pending.
  Production also needs its own database and public HTTPS webhook endpoint;
  do not reuse this sandbox database for live balances.

### PayPal

Set `PAYPAL_CLIENT_ID`, `PAYPAL_CLIENT_SECRET`, and `PAYPAL_WEBHOOK_ID` for
the same sandbox/live application. Subscribe to `PAYMENT.CAPTURE.COMPLETED`
at `https://YOUR_HOST/api/v1/webhooks/paypal` (a reachable HTTPS address).

The server creates an Orders v2 order. Return from approval triggers an owned,
idempotent server-side capture. A verified capture webhook also credits balance
if the user closes the return page. Webhook signatures are verified by PayPal's
verification endpoint. Approval alone never credits balance.

### Integrity and operation

The server selects package amounts and return URLs. It persists a payment before
requesting checkout, so a fast webhook cannot race an absent payment. A browser
idempotency key recovers checkout creation after a timeout. Payment event
verification, amount/currency/provider matching, ledger insertion and balance
update prevent forged callbacks and repeated credit. Browser redirects do not
change balances. The local inbox processor runs with the API; cloud profiles
require the existing inbox processor deployment.

Set `PAYMENT_MODE=live` only after real Firebase and payment sandbox verification
and connecting the real translation worker. Stripe key mode is checked; PayPal
uses the corresponding sandbox/live hostname. Missing provider credentials
disable that provider. Demo identities never enable payment gateways. Payments
require persistent storage; multi-replica deployment still needs a shared
transactional store instead of separate SQLite databases.

## Run and current verification limits

Copy `.env.example` to `.env` and fill the fields locally. Native
`python -m videotranslator.api` loads `.env`; Compose interpolates it via
`docker compose -f compose.chat.yml up --build`. Never commit keys or the service
account. Existing translation-provider keys remain private on the server.

Tests: `python -m pytest tests/unit`. Tests cover default pricing, quote-before-
charge, audio output format, frozen uploads, source changes, SQLite recovery,
concurrent confirmation, payment idempotency, unpaid callbacks, bad/stale
signatures, wrong amounts/providers, checkout timeout recovery and ownership.
Browser QA uses an isolated demo database and synthetic 90-second audio.

The localhost:8090 deployment now runs in a container with `local-full` and the
real Docker GPU engine connected; see [LOCAL_ENGINE.md](LOCAL_ENGINE.md).
Standalone `local-ui` still rejects translation before charging. No local
profile reports simulated work as successful or offers placeholder downloads.
Stripe sandbox checkout, callback verification and balance crediting are verified
as noted above; local live Stripe setup is described below; PayPal remains unconfigured. Firebase server
credentials and external Chrome Google sign-in are verified as noted above.
Docker Desktop is available at its per-user installation path; the Studio and
GPU services are running in containers.

Official integration references:
[Firebase Google](https://firebase.google.com/docs/auth/web/google-signin),
[Firebase email/password](https://firebase.google.com/docs/auth/web/password-auth),
[Stripe Checkout](https://docs.stripe.com/api/checkout/sessions),
[PayPal webhook verification](https://developer.paypal.com/api/rest/webhooks/rest/).

## Task history

The account bar opens Task history without discarding the current conversation.
Records come from the authenticated `/api/v1/jobs` endpoint and show newest
first, with queued, in-progress, error, completed, waiting, cancelled and expired
filters. The view refreshes every five seconds while visible. Details include
filename, media type, target language, quote, timestamps and progress; completed
outputs can be downloaded while available. Deleting or expiring media does not
remove its job record. Each retry remains a separate record linked to its source.
Failed task details distinguish returned account balance from cash refunds.
The header refreshes when a balance return is observed.
## Payment-return recovery (2026-09-23)

The local Stripe CLI listener must remain running to forward webhooks for the configured payment mode.
The return page can additionally POST to the owned payment's `/reconcile`
endpoint when a notification was missed. The server retrieves the stored Stripe
Checkout Session and verifies its ID, payment mode, paid/completed status,
application payment reference, currency and exact amount. It then uses the same
verified-event inbox and atomic purchase ledger as webhook delivery. Retries and
late webhooks credit the purchase once; return URL parameters are not proof of
payment and cannot select a different Stripe session.

While checking, the top-up dialog hides amount selection, payment providers and
refund terms. It displays confirmation status and an explicit no-repeat-payment
message. Success shows the balance and a Continue translation button; prolonged
or failed checks show Check payment again. Checks have bounded network waits and
polling, and the pending payment survives reloads for the same signed-in user.
This does not change Stripe sandbox/live mode or start translation automatically.

## Local live Stripe setup (2026-09-23)

The paired-environment workflow now supersedes manual key/file swapping:
use `stripe.ps1 sandbox`, `stripe.ps1 live` or `stripe.ps1 status` from the
repository root. See [STRIPE_ENVIRONMENTS.md](STRIPE_ENVIRONMENTS.md) for local
commands, isolated storage, rollback behavior and cloud secret configuration.

The daybreak live merchant uses a dedicated restricted key named
`VideoTranslator live backend`: Checkout Sessions write, Payment Intents read,
Events read, and Stripe CLI Debugging Tools write. No payout, refund, account
administration or general-purpose write permission is granted. Credentials stay
in ignored local configuration, never in source control.

For the existing local Docker deployment, `.env` selects `PAYMENT_MODE=live`,
`STUDIO_STORE_FILE=store-stripe-live.db` and
`STUDIO_QUEUE_FILE=inspection-queue-live.db`. The previous sandbox database,
task history and configuration backup are retained separately; sandbox credit
must not be imported into real balances. Firebase identities are unchanged and
new live wallets start at zero.

`deploy/stripe-listen.py` explicitly uses `stripe listen --live` for a live key
and refuses a mode/key mismatch or a non-local forwarding destination. Run it
with the project Python environment and the official Stripe CLI on PATH. It
saves the CLI signing secret to `.env`; recreate the Studio container if that
secret changes. Keep the listener running while accepting local payments;
restart it after restarting Windows. Checkout return reconciliation also
recovers a paid order whose webhook was missed, without double-crediting.

The gateway rejects events and API responses whose `livemode` does not match
the configured key. Live Checkout creation, retrieval and expiry were verified
with an unpaid USD 1 session. No actual charge was submitted by the setup check.
Payment completion and wallet credit still require a real user checkout.

This enables real payments on `http://localhost:8090`; it does not deploy a
public website. Public launch needs the actual HTTPS app domain, matching return
URLs, a durable public Stripe webhook endpoint and its signing secret, plus the
correct business website in Stripe. A temporary local CLI listener is not the
public deployment's webhook service.
