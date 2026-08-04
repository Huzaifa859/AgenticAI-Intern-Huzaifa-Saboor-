# Dockerfile
# ==========
#
# Jupyter environment for Codebase Assistant.
#
# Paired with docker-compose.yml, which also brings up an Ollama
# container so the full system (notebook + local llama3) runs with a
# single `docker compose up`, per the proposal's Deployment section.
#
# NOTE: Placeholder. This builds and starts Jupyter, but the image is
# not yet usable end-to-end -- see the TODOs below.

FROM python:3.11-slim

# git is required at runtime: the ingestion pipeline clones the target
# repository into the container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# TODO: codebase_assistant/requirements.txt currently pins only
# pydantic. Add the rest of the stack (jupyter, chromadb,
# sentence-transformers, GitPython, pyflakes, pytest, and an
# OpenRouter/Ollama client) before this image can run the notebook.
COPY codebase_assistant/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# TODO: source these from Config rather than duplicating defaults here.
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    OLLAMA_BASE_URL=http://ollama:11434

# Jupyter.
EXPOSE 8888

# TODO: token-based auth is disabled for local development only. Do not
# expose this container beyond localhost as configured.
CMD ["jupyter", "notebook", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token="]
