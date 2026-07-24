FROM python:3.11-slim-bookworm

# Verified against the official repository HEAD on 2026-07-22.
ARG INDEX_TTS_COMMIT=13495845e3028f0bb6ca1462ad22aa0e76349e40

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    INDEXTTS_PATH=/app/volumn/.cache/index-tts \
    INDEXTTS_MODEL_DIR=/app/volumn/.cache/index-tts/checkpoints \
    INDEXTTS_CONFIG_PATH=/app/volumn/.cache/index-tts/checkpoints/config.yaml

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /bin/

RUN git clone https://github.com/index-tts/index-tts.git /opt/index-tts \
    && cd /opt/index-tts \
    && git checkout "${INDEX_TTS_COMMIT}" \
    && uv sync --frozen --no-dev

ENV VIRTUAL_ENV=/opt/index-tts/.venv \
    PATH=/opt/index-tts/.venv/bin:$PATH \
    INDEXTTS_PATH=/opt/index-tts

COPY deploy/requirements-cloud.txt /tmp/requirements-cloud.txt
RUN uv pip install --python /opt/index-tts/.venv/bin/python \
        --index-strategy unsafe-best-match \
        -r /tmp/requirements-cloud.txt \
    && rm /tmp/requirements-cloud.txt

COPY src/volumn /app/volumn
COPY deploy/download-models.sh /app/deploy/download-models.sh
RUN chmod +x /app/deploy/download-models.sh \
    && mkdir -p /app/volumn/.cache/index-tts/checkpoints

WORKDIR /app/volumn

ENTRYPOINT ["/app/deploy/download-models.sh"]
CMD ["python", "translate_video.py", "--help"]
