# Cloud CPU and outbound GPU agents

## Deployed status (2026-09-29)

The Japan East CPU VM serves `https://vidyi.cc` with Studio, engine control, the
CPU engine worker, PostgreSQL and the GPU broker. One Windows RTX 4060 Ti host is
registered through outbound HTTPS. The Azure T4 Container App is an active fallback
provider with min 0/max 1. The public site reports processing available, Firebase
sign-in and the existing live payment mode. New job admissions are on.
The active fallback revision is `videotranslator-gpu--hybrid-1d77ff`; both workers
use engine digest `sha256:1d77ff2a16fcd3cd49d85769e7669bf0970444c1dc4a429e16ed7b12ec9ba7ad`.
The original `azure-jp-t4` profile identifier remains in Studio for compatibility;
GPU execution is selected by the new worker and broker configuration, not that
display identifier.

A synthetic audio check exercised remote tokenization, Demucs separation into
three stems and IndexTTS2 speech generation. All passed, with the broker reporting
one ready host. A full user video acceptance test is pending the user's sample.
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
another host claims it. More than one host can register, though the current cloud
engine runs one whole video at a time because it holds a deployment-wide advisory
lock. Multiple GPU hosts therefore provide failover and task assignment; they do
not yet translate multiple videos simultaneously.

## Provider priority and fallback

`GPU_PROVIDER_MODE=hybrid` makes the durable scheduler prefer an online outbound
GPU host. A job receives a permanent `gpu_provider` assignment when a worker claims
it. While any local host has sent a ready heartbeat within 30 seconds, unassigned
jobs are invisible to the Azure scaler and the CPU worker claims them for `local`.
When no local host is ready, queued unassigned jobs appear in `gpu_runnable_work`,
which wakes the Azure T4 worker and assigns them to `azure_t4`. An assigned Azure
job remains visible to the scaler until it is terminal, even if a local host returns.
This prevents duplicate execution and keeps Azure from scaling down mid-job.

The deployment-wide advisory lock still processes one whole video at a time. The
T4 provides availability when local GPUs are offline; it is not a parallel spillover
worker while a local video is already running. A provider failure after a job starts
fails that attempt closed, and a retry uses provider availability at retry time.

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

To enable Azure fallback after the local-only cutover, deploy the same immutable
engine image to both workers, set `GPU_PROVIDER=local` on the CPU engine worker and
`GPU_PROVIDER=azure_t4` on the Container App worker, then change
`GPU_PROVIDER_MODE=hybrid` and recreate engine control. Keep the T4 revision active
with min 0/max 1 and its existing `gpu_runnable_work` scaler.

If all home GPUs are offline before a job is claimed, Azure wakes and handles the
job. A host that loses a lease after its job starts stops publishing; that attempt
fails closed instead of moving mid-run between providers. The user can retry as a
new attempt. Large audio transfers use the home's upload connection and may affect
total processing time.

Only `/api/v1/gpu-workers/*` is publicly proxied to the broker. Keep its `/health`,
`/tokenize`, `/synthesize`, `/separations` and PostgreSQL ports on the private
Docker network. Revoke a compromised host by removing its token from
`GPU_WORKER_TOKENS` and recreating the broker.
