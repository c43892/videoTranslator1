# Stripe environment switching

## Local Windows deployment

Run from the repository root:

```powershell
.\stripe.ps1 status
.\stripe.ps1 sandbox
.\stripe.ps1 live
```

`test` is an alias for `sandbox`. `-CheckOnly` validates a proposed switch without
changing the selected mode, restarting services, or contacting Stripe.
The command requires this project's `.venv`, Docker Desktop and the official
Stripe CLI. It targets only the `videotranslator-studio` Compose project and its
own tracked local listener; GPU worker containers and other projects stay running.

Both credential pairs are stored locally in the ignored `.env`:

```
STRIPE_SANDBOX_SECRET_KEY=...
STRIPE_SANDBOX_WEBHOOK_SECRET=...
STRIPE_LIVE_SECRET_KEY=...
STRIPE_LIVE_WEBHOOK_SECRET=...
```

The original sandbox configuration backup and current live credentials were
used to initialize these fields. No secret was committed. Existing databases
are preserved: sandbox uses `store-stripe-sandbox.db` and `inspection-queue.db`;
live uses `store-stripe-live.db` and `inspection-queue-live.db`, all under
`vt-data/chat-preview`. The Compose configuration maps these files under `/data`.

The switch command validates the target, builds the Studio image, stops request
admission, rechecks unfinished work, verifies the target Stripe CLI connection,
updates `.env`, starts the matching webhook listener, recreates Studio and
checks its reported mode. A lock prevents two switches at once. Failures after
stopping Studio restore the previous `.env` and restart the previous mode.
The rollback copy is `.env.stripe-switch.rollback` (also ignored by Git).

Switches to a different mode are refused while there are queued/running/media
inspection tasks or unresolved top-ups. Finish verification of a pending payment
before switching. For a pending Stripe checkout the helper queries Stripe;
an expired, unpaid session does not block switching. Open, processing, paid but
uncredited, or unverifiable payments block switching. This helper does not
cancel or refund payments.
Waiting-for-credit drafts and completed tasks remain in their original database.
Switching back restores that environment's balance and task history.

Refresh the browser after switching. Sandbox displays a persistent Test payments
badge. Conversation/payment browser storage is separated by mode, and an old tab
that supplies the previous mode receives `payment_environment_changed` rather
than silently creating an order in a different mode. API clients should send
`X-Payment-Mode: sandbox` or `live` to receive the same protection.

Use `stripe.ps1 live` (or `sandbox`) after restarting Windows to restart the local
listener as well. `status` reports configured and running modes separately.

## Cloud deployment

The same application selection logic is used on cloud hosts. Configure the
fields in `deploy/stripe-cloud.env.example` using the cloud service's environment
settings and secret references. Do not bake the file or secrets into an image.

1. Supply a sandbox key and a live key for the intended merchant/environment.
2. In each Stripe environment, configure the public HTTPS endpoint
   `https://YOUR_DOMAIN/api/v1/webhooks/stripe` for
   `checkout.session.completed` and `checkout.session.async_payment_succeeded`.
3. Put each endpoint's signing secret in the matching `STRIPE_*_WEBHOOK_SECRET`.
   These are cloud endpoint secrets, **not the local Stripe CLI secrets**.
4. Mount durable storage. Configure distinct `PAYMENT_SANDBOX_STORE_PATH` and
   `PAYMENT_LIVE_STORE_PATH`, and distinct `PAYMENT_*_QUEUE_PATH` values. The
   process selects the matching paths using `PAYMENT_MODE`.
5. Set `PUBLIC_APP_URL` to the actual HTTPS application URL; start with
   `PAYMENT_MODE=sandbox`, then verify checkout, callback crediting, bonuses and
   translation before selecting `PAYMENT_MODE=live` and restarting/redeploying.

Once both profiles are configured, `PAYMENT_MODE` is the only setting needed to
select one. If one profile key is present, the selected profile must be complete;
the app never silently falls back to the legacy `STRIPE_SECRET_KEY`. A database
records its payment mode and refuses startup under the other mode. Stripe API
responses and webhook events must also match the selected key's live/test mode.

For a same-URL cloud cutover, stop accepting new jobs/checkouts, drain active work
and resolve pending payments first. Stop the old revision before activating the
new one: do not distribute requests between sandbox and live replicas under the
same URL. Roll back by selecting the previous mode and its existing storage.
Keep this SQLite deployment to one Studio replica. A future multi-replica setup
needs a shared transactional datastore; changing Stripe mode does not provide it.
An alternative is separate staging/production apps, each with its own URL,
storage and Stripe webhook configuration, which avoids interrupting production.

`stripe.ps1` manages the local deployment only. It does not deploy Azure resources
or edit cloud secret references. Cloud infrastructure rollout remains a separate
task. PayPal is not switched by this Stripe-only helper and should remain
unconfigured until its sandbox/live credentials are separately managed.

References: [Stripe sandbox testing](https://docs.stripe.com/testing),
[Stripe webhooks](https://docs.stripe.com/webhooks),
[Stripe secret handling](https://docs.stripe.com/keys-best-practices).

## vidyi.cc live cutover — 2026-09-24

The public Azure site now selects `PAYMENT_MODE=live` through its private Compose
`.env`. The tracked Compose file uses `${PAYMENT_MODE:-sandbox}`. Local Windows
remains in sandbox mode. Application images, GPU admission and limits are unchanged.

The existing restricted live key was reused without expanding its permissions.
Stripe destination `vidyi-production` (`we_1UJ0xcA9Wq7eEimkOpDsynTa`) is active at
`https://vidyi.cc/api/v1/webhooks/stripe`, listening for
`checkout.session.completed` and `checkout.session.async_payment_succeeded`.
The destination-specific signing secret is configured privately on the VM; the
local Stripe CLI signing secret is not reused for this endpoint.

Before cutover, Studio was stopped and its sandbox database rechecked: no unfinished
translations/downloads; the only pending checkout was verified expired and unpaid.
Sandbox SQLite was backed up and retained. Live uses its own initially empty
wallet/history and inspection databases. Sandbox balances and test results were
not copied into live balances or history. Existing Firebase login continues to work.

Validation confirmed public live mode, zero initial live balance, rejection of
stale sandbox headers, and idempotent creation of a real-mode USD 1.00 Checkout
session with the correct vidyi.cc return URL. That validation session was expired
without payment. A deliberately ignored non-payment payload signed with the new
endpoint secret was accepted; invalid signatures were rejected. No actual charge
or genuine Stripe payment-success delivery was performed in this cutover test.
Browser verification confirms the test badge is absent and Stripe top-up is enabled.

VM configuration backups are `.env.before-stripe-live-20260924`,
`studio.env.before-stripe-live-20260924` and
`compose.cpu.yml.before-stripe-live-20260924` under `/srv/videotranslator`.
Switching back later requires draining and checking **live** tasks/payments first;
never blindly restore a sandbox config while live payments remain unresolved.

The original 05:16 UTC shutdown was retained during payment cutover. In the
subsequent explicit request for continuous web service on 2026-09-24, the user
authorized keeping the CPU VM online, and that shutdown schedule was disabled.
GPU remains task-driven with min=0/max=1; idle verification found zero replicas.
CPU and storage costs continue independently of GPU usage. The home downloader
still needs its Windows host online to service YouTube downloads.
