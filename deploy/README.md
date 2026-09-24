# VideoTranslator cloud image

> Historical deployment path: this document describes the older single-image Azure ML workflow. For the current stable Studio/engine migration to Japan East Container Apps T4, use [the migration plan](../docs/AZURE_JAPAN_T4_MIGRATION_PLAN.md) and [resource verification record](../docs/AZURE_JAPAN_RESOURCES.md). The image, local-Whisper default, East US quota status, and “no resources created” statement below are historical; they are not the current deployment baseline.

The cloud image is stored in GitHub Container Registry (GHCR), not Azure Container Registry:

```text
ghcr.io/c43892/videotranslator@sha256:f4202b32423eebe3077be096e480da5d00bbc8dd11632198914b9686ec31127e
```

The image contains the application and Python/CUDA dependencies. IndexTTS2 model weights are downloaded into the job's temporary disk when the container starts, so there is no idle Azure storage resource to maintain.

## Current state

- Docker image build: passed
- Python imports: passed
- NVIDIA/CUDA visibility: passed on the local RTX 4060
- FFmpeg video/audio split smoke test: passed
- T4 quota request: 4 vCPUs requested in East US; request `624f0613-0ef6-429a-ba4d-915aecffcd84` is in progress
- Full T4 workflow: waiting for that quota request to be approved
- Azure resources created during preparation: none

## Check quota (read-only)

From PowerShell:

```powershell
.\deploy\azure\check-t4-quota.ps1
```

A successful check means an `NC4as_T4_v3` job can be submitted. An error means the quota is still below four available vCPUs. This check does not create resources and does not incur Azure charges.

## Current processing choices

- Transcription: local `openai-whisper==20250625`, model `turbo`
- Translation: official DeepSeek API, `deepseek-v4-flash` with thinking disabled
- Source separation: local `demucs==4.1.0`, model `htdemucs`
- Voice cloning: local IndexTTS2 pinned to the current official repository HEAD
- Object storage: Azure Blob through `az://container/blob/path` URIs

The only required inference API is DeepSeek. `DEEPSEEK_API_KEY` must be supplied
as a secret at job runtime. The pipeline no longer calls the external DFN speech
separation service and it ignores embedded/provided subtitles.

For a direct Azure Blob job, set either `AZURE_STORAGE_CONNECTION_STRING` or
`AZURE_STORAGE_ACCOUNT_URL` (managed identity), then run:

```text
python process_cloud_job.py \
  --input-uri az://media/in/input.mp4 \
  --output-uri az://media/out/input-translated.mp4 \
  --target-lang English \
  --transcription local
```

## Run once after quota is approved

The one-shot validation accepts only a source video/audio file and target language.
It performs the complete local transcription, separation, voice cloning, mixing,
and rendering path; only translation is sent to DeepSeek.

```powershell
.\deploy\azure\run-once.ps1 `
  -InputVideo C:\path\input.mp4 `
  -TargetLanguage English
```

The script:

1. refuses to continue until the T4 quota is available;
2. creates a uniquely named temporary resource group and Azure ML workspace;
3. registers the pinned GHCR image and submits a serverless T4 job;
4. waits for completion and downloads the output;
5. deletes the whole temporary resource group in a `finally` block.

Do not pass `-KeepAzureResources` for normal occasional use. It is only for debugging and intentionally disables automatic cleanup.

The image must be publicly readable before Azure ML can pull it without registry credentials. The GHCR package is currently private pending explicit approval to change its visibility.

## Cost behavior

Before the script creates the resource group, Azure cost is zero. During a run, the serverless GPU is billed for job execution and provisioning time. The cleanup step removes the workspace and its supporting resources after the output is downloaded. The public GHCR image and local Docker image can remain available between runs without keeping an Azure VM running.
