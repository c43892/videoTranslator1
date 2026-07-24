#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${INDEXTTS_MODEL_DIR:-/app/volumn/.cache/index-tts/checkpoints}"

if [[ "${VIDEOTRANSLATOR_SKIP_MODEL_DOWNLOAD:-0}" != "1" ]] \
  && [[ ! -f "${MODEL_DIR}/config.yaml" ]]; then
  echo "Downloading official IndexTTS2 checkpoints to ${MODEL_DIR}..."
  mkdir -p "${MODEL_DIR}"
  hf download IndexTeam/IndexTTS-2 --local-dir "${MODEL_DIR}"
fi

exec "$@"
