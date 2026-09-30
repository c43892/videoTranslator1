# Cloud CPU and outbound GPU agents

## Deployed status (2026-09-29)

The Japan East CPU VM serves `https://vidyi.cc` with Studio, engine control, the
CPU engine worker, PostgreSQL and the GPU broker. One Windows RTX 4060 Ti host is
registered through outbound HTTPS. The Azure T4 Container App is an active fallback
provider with min 0/max 1. The public site reports processing available, Firebase
sign-in and the existing live payment mode. New job admissions are on.
The active fallback revision is `videotranslator-gpu--pipeline-1790714723`.
The CPU engine control, CPU worker, GPU broker and Azure T4 worker use engine
digest `sha256:9ecd14d445f1f61c5475e60098ac3a5be3ec00d9fdc6b1f060de7b25207a44ba`;
Studio uses control digest
`sha256:978ce4ceb73f172f6ac27499b85195ccc8418d6f8207fa1c6f2ce788c4799796`.
The original `azure-jp-t4` profile identifier remains in Studio for compatibility;
GPU execution is selected by the new worker and broker configuration, not that
display identifier.

A synthetic audio check exercised remote tokenization, Demucs separation into
three stems and IndexTTS2 speech generation. All passed, with the broker reporting
one ready host. A 19.59-minute user video subsequently completed on the local
provider and a second one-minute job completed after the ordered scheduler cutover.
The Windows host is set to stay awake on AC power and Docker Desktop is configured
to start when the user signs in. Docker Desktop does not provide GPU service before
Windows sign-in.

The public Studio, PostgreSQL, engine control, YouTube import coordination,
transcription, translation, alignment, mixing, billing and result publication run
on the cloud CPU VM. The existing YouTube download agent remains a separate
outbound service. `gpu-broker` also runs on the CPU VM. Each GPU host runs only IndexTTS2/Demucs and an
outbound HTTPS agent. No inbound port, Azure credentials, provider API key or payment
credential is needed on a GPU host.

```text
browser -> Caddy -> Studio -> engine-control -> PostgreSQL
                                  |                  |
                             Azure Files <---- CPU engine-worker
                                  |                  |
                                  +---- gpu-broker <-+  (private Docker network)
                                          ^
                                          | HTTPS heartbeat, claim, media transfer
                                  GPU agent(s) -> local IndexTTS2/Demucs
```

Each host runs both supported GPU operations and has a unique random token. The broker accepts a comma-separated token
list and persists workers and tasks in PostgreSQL. A ready host claims one task at a
time. A task lease lasts 45 seconds and is renewed by heartbeats every 5 seconds.
Expired leases return to the queue; a former owner cannot publish a result after
another host claims it. More than one host can register. Registered agents behind
the same `local` provider currently form one provider slot and provide failover;
independent concurrent GPU services should receive distinct provider names in the
ordered provider list.

## Provider priority and fallback

`GPU_PROVIDER_MODE=hybrid` uses one durable FIFO queue and an explicit finite
provider order. The production defaults are `GPU_PROVIDER_PRIORITY=local,azure_t4`
and `GPU_PROVIDER_CAPACITIES=local=1,azure_t4=1`. The scheduler examines providers
in that order for the oldest unassigned job: it assigns local when a recent ready
agent exists and the local slot is free; otherwise it assigns Azure T4 when its slot
is free. When all slots are busy or offline, the job remains unassigned and workers
check again every `GPU_SCHEDULER_POLL_SECONDS=10` seconds. The T4 KEDA trigger also
uses a 10-second polling interval, so a scale-to-zero worker follows the same cadence.

Each provider has a stable, separate PostgreSQL advisory lock, so one local video
and one T4 video may run concurrently. A job receives a permanent `gpu_provider`
assignment when claimed. Assigned Azure work remains visible to the scaler until
terminal, preventing scale-down mid-job. Provider selection is never changed during
an attempt: a provider failure fails that attempt closed and a user retry re-enters
the ordered queue. To add another independent GPU service, give its worker a unique
provider name, insert that name into `GPU_PROVIDER_PRIORITY`, set its capacity, and
configure either a provider heartbeat or scale-to-zero availability. Do not assign
capacity greater than the number of independently running workers for that provider.

## Private configuration

Generate a distinct 32+ character random secret for each GPU host and another for
the internal broker. Keep them in private env files, not in Git or the browser:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

On the cloud VM, add `GPU_BROKER_INTERNAL_TOKEN` to the Compose `.env`. Create
`gpu-broker.env` with `DATABASE_URL` set to the **same PostgreSQL database** used
by `engine.env`, and `GPU_WORKER_TOKENS` set to the comma-separated host tokens.
The broker receives no Stripe, Firebase, Azure or translation provider keys. Keep
the existing `studio.env`, `engine.env`, `postgres.env`, payment mode and YouTube
worker configuration. The engine worker receives the internal broker token through
Compose and uses the broker URLs for its two GPU operations.

On each GPU host, copy `.env.home-gpu.example` to a private `.env.home-gpu`, set
`GPU_WORKER_TOKEN` to that host's token, and set `GPU_AGENT_SERVER_URL` to the
public HTTPS site. `GPU_TASK_VOLUME` and `GPU_MODEL_VOLUME` identify local Docker
volumes; on the original Windows host their defaults reuse the existing task and
model volumes, avoiding a model download. On another host, use unique volume names,
create both named volumes with `docker volume create`, then allow the TTS container
to load its models before enabling work. The host needs
Docker Desktop, WSL2 NVIDIA GPU support, and enough free disk for transferred audio.
Do not run two IndexTTS2 containers on the same 4060 Ti simultaneously.
Enable Docker Desktop's sign-in auto-start on Windows and keep the host awake;
`restart: unless-stopped` resumes the containers when Docker starts, but it cannot
start Docker Desktop after a user logout or computer sleep.

```powershell
docker compose -f compose.home-gpu.yml --env-file .env.home-gpu config --quiet
docker compose -f compose.home-gpu.yml --env-file .env.home-gpu up -d --no-build
```

Run `up` on the original Windows host only during cutover, after its old TTS
container has stopped. The old local stack must retain its Docker volumes.
The `--no-build` command expects locally built `videotranslator-local-tts:latest`
and `videotranslator-home-gpu-agent:latest` images. For a new host, build them with
`docker compose -f compose.home-gpu.yml --env-file .env.home-gpu build` first.
The agent is deliberately HTTPS-only. It exposes no listening port.

## Live cutover

1. Build and push a new immutable **engine** image with `gpu_broker.py`, and stage
   the updated Caddyfile/Compose on the CPU VM. Keep the current control image,
   payment configuration and databases. Back up the private env files.
2. Temporarily disable new admissions (`GPU_STARTS_ENABLED=false`,
   `CLOUD_ACCEPT_JOBS=false`) and let any queued or active Azure T4 job finish.
   Confirm the PostgreSQL `gpu_runnable_work` view has no rows before starting the
   CPU engine worker. This prevents simultaneous old and new execution.
3. Set `REVERSE_GPU_ENABLED=true` and `GPU_PROVIDER_MODE=local_only` in the CPU VM's private Compose `.env` and
   recreate `engine-control` with the new engine image. It changes the existing
   `gpu_runnable_work` view to report no T4 work, so the Container App's existing
   zero-minimum scaler returns to zero without Azure control-plane credentials.
   Verify the view has zero rows and the T4 is idle. If Azure management access is
   available, also set its maximum replicas to zero or deactivate the revision.
   Keep the resource
   available for rollback until the new deployment has passed the user's video
   test.
4. Start the broker explicitly with `docker compose -f compose.cpu.yml
   --profile reverse-gpu up -d gpu-broker` and enable the Caddy route. The GPU
   services are behind a Compose profile so a routine `up -d` during staging
   cannot start the CPU engine worker alongside the T4. On the original Windows
   host, stop the old
   full local Compose stack after its jobs finish, preserving its named volumes;
   do not use `down -v`. Start the home GPU Compose stack. Confirm its heartbeat
   from broker health (`workers_online >= 1`). Start the CPU engine worker with
   `docker compose -f compose.cpu.yml --profile reverse-gpu up -d engine-worker`. Submit
   a small **synthetic** GPU task to check transfer, lease renewal and output
   publication, then re-enable admissions. The user will supply the actual video
   acceptance test later.
5. Check `https://vidyi.cc/api/v1/health/ready`, the private engine-control health,
   broker `/health`, GPU agent logs, and the active web payment mode. Keep live
   billing settings and existing customer prices intact during this migration.

Rollback: disable new admissions, stop the CPU engine worker after its current
job finishes, restore the prior Caddy/Compose and engine image, reactivate the
T4 revision/scaler, set `REVERSE_GPU_ENABLED=false` and recreate engine-control to
restore the scaler view, verify it has capacity, and re-enable admissions. Never allow
both engine workers to execute against the same database at once.

To enable ordered Azure spillover after the local-only cutover, deploy the same immutable
engine image to both workers, set `GPU_PROVIDER=local` on the CPU engine worker and
`GPU_PROVIDER=azure_t4` on the Container App worker, then change
`GPU_PROVIDER_MODE=hybrid` and recreate engine control. Keep the T4 revision active
with min 0/max 1 and its existing `gpu_runnable_work` scaler. The scaler view exposes
unassigned work when every higher-priority provider is offline or at capacity, not
only when local is offline.

If all home GPUs are offline or the local provider slot is busy before a job is
claimed, Azure wakes and handles the next FIFO job. A host that loses a lease after
its job starts stops publishing; that attempt
fails closed instead of moving mid-run between providers. The user can retry as a
new attempt. Large audio transfers use the home's upload connection and may affect
total processing time.

## Ordered scheduler production verification (2026-09-29)

The guarded rollout completed without interrupting the pre-existing local job.
Production was verified with `GPU_PROVIDER_PRIORITY=local,azure_t4`, capacities
`local=1,azure_t4=1`, and a 10-second scheduler interval. The Azure revision is
active with the same engine digest, `GPU_PROVIDER=azure_t4`, min 0/max 1, a
10-second KEDA polling interval, and zero replicas while idle. The CPU worker and
broker use the new engine digest, the private engine-control health check and
`https://vidyi.cc/api/v1/health/ready` returned HTTP 200, admissions were restored,
one ready local agent had a current heartbeat, and `gpu_runnable_work` was empty.

The one-shot verifier initially failed after the deployment had already completed:
first it inspected a container as though it were an image and requested the absent
`RepoDigests` field; a retry then encountered an expired ACR pull authorization even
though the required immutable image was already present and running. The verifier
was corrected to inspect the container's configured image and to validate the
already-running digest without an unnecessary registry pull. Its final run emitted
`ORDERED_GPU_ROLLOUT_COMPLETE` and exited successfully.

## Stage-progress UI release (2026-09-29)

Commit `551d147` added a compact current-stage line beneath the progress bar in
both the live job card and task-history detail view. It uses the engine's existing
stage and percentage fields, includes English and Simplified Chinese labels, and
does not change scheduling or GPU execution. That release used immutable control
digest
`sha256:20020322e89cee19377902ede13147f635745bda5ae0122eb42ee89322335374`.
Only the `web` container was recreated; engine control and GPU services were left
running. The private and public readiness checks returned HTTP 200, and the public
HTML and versioned JavaScript assets were verified to contain the new release.

## Public video-link import release (2026-09-29)

Commit `bcddfc5` changed the UI to describe a generic video link without
advertising additional source sites. Studio admits safe public HTTPS URLs instead
of rejecting every non-YouTube host. The outbound Windows download agent decides
actual compatibility using yt-dlp's site-specific extractors; the generic extractor
is disabled so arbitrary webpages cannot be used to reach the home network. Private
and local addresses, URL credentials and non-default ports remain blocked. Failed,
private, unavailable and region-restricted sources return the existing import-failed
state with a generic unsupported-or-unavailable message.

Production Studio uses immutable control digest
`sha256:978ce4ceb73f172f6ac27499b85195ccc8418d6f8207fa1c6f2ce788c4799796`.
Only the `web` container was recreated. The private and public readiness endpoints
returned HTTP 200, the public assets contained only generic link wording, and the
production container accepted and normalized a non-YouTube public URL in a read-only
smoke check. No download or GPU job was submitted for verification.

## Local GPU performance investigation (2026-09-29)

The completed 19.59-minute video spent 13 minutes 49.7 seconds waiting before its
execution lease and 1 hour 59 minutes 19.1 seconds executing. The queue delay is not
GPU processing time. Within execution, the measured/inferred stage windows were:

| Window | Wall time | Evidence and limitation |
| --- | ---: | --- |
| Execution start to separation task creation | 10.2 s | Database timestamps |
| Demucs task creation to first tokenizer task | 6 min 42.0 s | Includes Demucs plus its downloads/uploads and CPU handoff |
| Tokenizer window to first synthesis task | 53.6 s | Five serial tokenizer tasks |
| First synthesis task to execution finish | 1 h 51 min 33.3 s | Includes the last synthesis, alignment/mix/encode and final publication |

Synthesis was the dominant window: 471 calls handled only 3,884 translated
characters, or 8.25 characters per call. The pipeline creates and waits for one
segment at a time, while the agent permits only one lease. Across the 470 measurable
creation-to-next-creation intervals, total time was 1 hour 49 minutes 37.4 seconds;
the mean was 13.994 seconds, median 12.415 seconds, p95 25.474 seconds and maximum
34.464 seconds. A simple latency regression was `7.848 seconds + 0.744 seconds per
character` (R-squared 0.719). This associates about 61.5 minutes with per-call fixed
cost across those intervals, although current telemetry cannot split that fixed
cost exactly among HTTPS/DB coordination, local service overhead and model inference.

The media was small as a delivered video but not as GPU intermediate data. Demucs
downloaded 224.5 MB and uploaded three 414.6 MB WAV stems, about 1.47 GB in total.
Synthesis downloaded 165.5 MB of reference WAVs and uploaded 38.9 MB of generated
audio, about 204.5 MB in total. More importantly, every one of the 471 synthesis
calls performs two reference downloads and one output upload, producing 1,413
audio transfers plus claim, heartbeat and completion requests. Bandwidth can
therefore contribute materially to the 6-minute-42-second Demucs window, while the
synthesis measurements point mainly to multiplied per-request latency and strictly
serial fine-grained inference rather than total byte volume.

All one separation task, five tokenizer tasks and 471 synthesis tasks finished in
`completed` state under one worker; there were no terminal failed or cancelled GPU
task rows. The schema does not preserve attempt-level history, so it cannot exclude
short lease retries. It also retains only a cumulative five-second GPU-activity
counter, not timestamped utilization, memory, clock or power samples. Consequently,
this run proves that orchestration granularity dominated end-to-end time but does
not prove that the RTX 4060 Ti's raw inference throughput is lower than a T4. There
is no same-video, same-revision T4 run in the current telemetry; the historical
18-second T4 validation included cold start and is not a valid comparison.

The smallest next-run instrumentation is to persist task completion timestamps and
agent-side monotonic durations for input download, local API wait/inference and
output upload, plus one-second `nvidia-smi` utilization/memory/power samples labeled
by job and task. A controlled comparison should reuse the same cached model, media,
segmentation and revision on both GPUs. Likely optimization candidates, pending
separate authorization, are batching/coalescing very short TTS segments, avoiding
the duplicate speaker/emotion reference download when both keys are identical, and
reusing reference audio across consecutive segments.

Only `/api/v1/gpu-workers/*` is publicly proxied to the broker. Keep its `/health`,
`/tokenize`, `/synthesize`, `/separations` and PostgreSQL ports on the private
Docker network. Revoke a compromised host by removing its token from
`GPU_WORKER_TOKENS` and recreating the broker.
