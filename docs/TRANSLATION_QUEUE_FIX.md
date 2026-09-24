# Automatic translation queue — 2026-09-24

## Cause and behavior

The reported queue-full task was `job_34f0bb2f718c3341`. At diagnosis there were no
active translations and the backlog counter was zero. GPU admission remained
disabled after the attended Azure validation (`GPU_STARTS_ENABLED=false`,
`CLOUD_ACCEPT_JOBS=false`). The engine also retained a cumulative validation limit
of one attempt despite historical executions already exceeding that count.

The UI mapped every admission block to queue-full, and reconciliation did not
revisit `awaiting_capacity`. Both are corrected. Confirmed waiting tasks persist
across restarts, are checked every 10 seconds in creation order, and automatically
advance when eligible. The Azure deployment admits one GPU task at a time.
Waiting for a slot does not debit points; admission atomically debits once and
creates the durable submission. Waiting tasks can be cancelled. Paused service or
exhausted application budget displays a saved-task/pause message instead of busy.
Insufficient account points still require topping up and confirming Start.

The initial cost-budget check now calculates the real maximum runtime before
checking the reservation; previously that first check could see zero seconds.

## Deployment

- Control: `vtranslatorjpe43892.azurecr.io/videotranslator/control@sha256:3b3d86be9cb81af6148c133409497f30c927440ce9c49f3787146a8a8195438b`
- CPU engine control: `vtranslatorjpe43892.azurecr.io/videotranslator/engine@sha256:be2b41cddfad4182b8d9ad93711fea6b7e48a8c115756a3ebda8442e65e94f12`
- Private `.env`: `GPU_STARTS_ENABLED=true`, `CLOUD_ACCEPT_JOBS=true`,
  `CLOUD_VALIDATION_MAX_JOBS=0`.
- Existing GPU revision/model image unchanged; replicas remain min=0/max=1.
- `MAX_ACTIVE_GPU_JOBS=1`, daily US$5/monthly US$10 application admission budgets,
  and engine maximum 1800 seconds retained. These estimates do not cap the entire
  Azure invoice (cold starts, cooldown, CPU, disks, storage, registry, etc.).
- Existing 05:16 UTC CPU auto-shutdown remains. This is still a budget-limited
  deployment, not an indefinitely online production operating plan.
- Stripe remains in sandbox mode; home download worker remains running.
- Remote `.env`, compose and SQLite backups use the suffix
  `before-auto-queue-20260924` under `/srv/videotranslator`.

## Verification

204 control Python unit tests, 18 frontend tests, and 7 private engine-control
tests passed. Coverage includes automatic resumption, one active GPU slot,
creation-order scheduling, cancellation without charge, concurrent duplicate
admission protection, and first-attempt budget reservation.

After deployment, public homepage and readiness returned HTTP 200. The existing
user task automatically moved from `awaiting_capacity` to `provisioning`, with
one charge ledger entry, without another user Start request. Azure scaled the
existing GPU revision from zero to one replica. Final media completion is a
separate pipeline check; provisioning alone is not proof of completed translation.

## Follow-up: cloud result playback

The user task subsequently completed successfully. The completed conversation card
hid the Play result button behind `config.local`; that guard is now removed.
The player sets anonymous CORS before assigning its signed media URL so remote
subtitle tracks can load, and supports inline playback. The page asset version is
`20260924-cloud-preview`.

Current control image:
`vtranslatorjpe43892.azurecr.io/videotranslator/control@sha256:1a446f05746106026fb395efe1286327486d7fe7688ef7fded7f6084572011ac`.
Only `app.js` and `index.html` are overlaid on the queue-fix image; engine, GPU,
budgets and admission settings are unchanged. The prior private `.env` is backed
up as `.env.before-cloud-preview-20260924`.

Verified the published JavaScript matches the local source, the completed card
shows Play result, and Chrome plays the existing user result: 1920×1080,
71.872 seconds, readyState=4, paused=false, currentTime=26.65 seconds, no media error.
This playback check did not start another translation.
