FROM python:3.12-slim-bookworm
RUN pip install --no-cache-dir httpx==0.28.1 \
    && useradd --uid 10001 --create-home agent && mkdir /data && chown agent:agent /data
COPY deploy/home-gpu-worker/agent.py /opt/gpu-agent/agent.py
USER 10001
ENV PYTHONUNBUFFERED=1 GPU_AGENT_DATA_ROOT=/data
CMD ["python", "/opt/gpu-agent/agent.py"]
