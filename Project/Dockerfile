# Dockerfile
# ==========
#
# Streamlit Web UI for Codebase Assistant (OpenRouter).
# Ships only the runtime app package + demo repo — not tests/docs/benchmarks.
#
# Build / run from the Project directory:
#
#     docker compose up --build
#     # open http://localhost:8501
#
# Secrets come from Compose env_file (.env) — never bake OPENROUTER_API_KEY
# into the image. See .dockerignore.

FROM python:3.11-slim

# git: clone GitHub repos at runtime. curl: Compose healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer first so code-only changes reuse the pip cache.
# Longer timeout/retries: PyPI downloads from inside Docker are often slow.
COPY requirements-docker.txt ./requirements.txt
ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --retries 10 --default-timeout=100 -r requirements.txt

# Runtime application only (see .dockerignore for excluded paths).
COPY app ./app
COPY codebase_assistant ./codebase_assistant
COPY .streamlit ./.streamlit
COPY examples/demo_repo ./examples/demo_repo
COPY .env.example ./.env.example

RUN useradd -m appuser \
    && mkdir -p /data/chroma /data/memory /data/history \
    && chown -R appuser:appuser /app /data

USER appuser

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    CODEBASE_ASSISTANT_DATA_DIR=/data \
    CHROMA_PERSIST_DIR=/data/chroma \
    MEMORY_STORE_PATH=/data/memory \
    RUN_HISTORY_PATH=/data/history/ui_run_history.jsonl \
    CA_WORKER_HOST=127.0.0.1 \
    CA_WORKER_PORT=8765 \
    CA_WORKER_URL=http://127.0.0.1:8765

EXPOSE 8501

# Warm worker on localhost:8765, then Streamlit on :8501 (only :8501 published).
CMD ["python", "app/docker_entrypoint.py"]
