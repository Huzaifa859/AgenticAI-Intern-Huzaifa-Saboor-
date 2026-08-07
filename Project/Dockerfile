# Dockerfile
# ==========
#
# Streamlit Web UI for Codebase Assistant (OpenRouter).
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
COPY codebase_assistant/requirements.txt ./requirements.txt
ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --retries 10 --default-timeout=100 -r requirements.txt

COPY . .

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
    RUN_HISTORY_PATH=/data/history/ui_run_history.jsonl

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]
