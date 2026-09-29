# Local RTX GPU service

This setup runs the Studio, private engine control and worker, PostgreSQL, and
IndexTTS2/Demucs GPU service from the source in this repository. It replaces the
older local setup in `LOCAL_ENGINE.md`, which depends on a branch and base images
that are not present in this clone.

## Prerequisites

- Windows 11, WSL 2, Docker Desktop with the WSL 2 backend, and an NVIDIA driver
  with WSL GPU support. Docker's GPU support on Windows requires the WSL 2 backend.
- An NVIDIA GPU with enough VRAM for IndexTTS2. The included TTS image downloads
  its model weights on first start and stores them in the `model-data` volume.
- The current pipeline uses OpenAI for speech transcription and DeepSeek for
  translation. Set `OPENAI_API_KEY` and `DEEPSEEK_API_KEY` in the ignored
  `.env.local-gpu` file to process real jobs.
- To use Firebase sign-in, set `AUTH_MODE=firebase`, the four `FIREBASE_*` web
  values, and save the admin credential as ignored `secrets/firebase-admin.json`.
  For local Stripe test payments, set `PAYMENT_MODE=sandbox` and the two
  `STRIPE_SANDBOX_*` values. Live payment credentials are not used locally.

On this PC, Docker Desktop is installed at
`C:\Users\c4389\Documents\DockerDesktop`; its CLI path is saved in the user
`PATH`. The usual per-user AppData installation was not visible from WSL here.

## Start

After Windows restarts to finish WSL 2 setup, start Docker Desktop. From the
repository root in PowerShell, build and start only the GPU service first:

```powershell
if (-not (Test-Path .env.local-gpu)) { Copy-Item .env.local-gpu.example .env.local-gpu }
# Generate unique POSTGRES_PASSWORD, ENGINE_CONTROL_TOKEN (32+ characters),
# and LOCAL_STORAGE_SIGNING_SECRET (32+ characters), then add provider keys.
docker compose --env-file .env.local-gpu -f compose.local-gpu.yml up -d --build --no-deps tts
docker compose --env-file .env.local-gpu -f compose.local-gpu.yml exec -T tts python -c "import torch; print(torch.cuda.get_device_name(0))"
docker compose --env-file .env.local-gpu -f compose.local-gpu.yml ps
```

The standalone GPU service does not use port 8090, so the native preview can
continue running during this check. The service loads IndexTTS2 on CUDA and runs
Demucs on the same GPU. Its health endpoint becomes ready after model download.

When the GPU service is ready, set the provider keys in `.env.local-gpu`, stop
the native preview, and start the rest of the stack:

```powershell
docker compose --env-file .env.local-gpu -f compose.local-gpu.yml up -d --build
```

The Studio is at `http://localhost:8090`. Only this port is published, and it is
bound to localhost. The private engine and GPU service stay on the Compose
network. Studio sign-in and payments follow `AUTH_MODE` and `PAYMENT_MODE` in
`.env.local-gpu`; the example defaults to local demo identity with payments off.

YouTube links are downloaded by Studio's background importer with `yt-dlp` in
`direct` mode. This local stack has no separate download worker. The optional
`home-worker` mode in `docs/HOME_DOWNLOAD_WORKER.md` is for a remote site that
needs an outbound polling downloader. Give Studio one video URL per job; playlist
URLs are not accepted as a translation source.

Keep `CLOUD_ACCEPT_JOBS=false` and `GPU_STARTS_ENABLED=false` until the GPU service
is healthy and both provider keys are configured. Then set both to `true` and
recreate `engine-control`, `engine-worker`, and `studio`.

```powershell
docker compose --env-file .env.local-gpu -f compose.local-gpu.yml ps
docker compose --env-file .env.local-gpu -f compose.local-gpu.yml up -d --force-recreate engine-control engine-worker studio
```

The control service can be healthy while the first IndexTTS2 model download is
still underway. Check `docker compose --env-file .env.local-gpu -f compose.local-gpu.yml logs -f tts`
for progress. The first image build and model download require substantial disk
space and network transfer.

The local engine control allows up to two hours per job. Its default 30-minute
cap is too short for longer videos with many IndexTTS2 speech segments; Studio
still applies each job's own duration-based runtime limit.

The native Python preview also uses port 8090. Stop it before starting Compose.
Studio data remains in ignored `vt-data/`; PostgreSQL, engine media, and model
weights persist in Docker named volumes.
