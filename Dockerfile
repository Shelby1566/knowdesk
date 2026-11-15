FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
RUN git clone https://github.com/sierra-research/tau2-bench.git /tau2-bench && git -C /tau2-bench checkout 033bf08
COPY pyproject.toml uv.lock ./
COPY kb ./kb
COPY agent ./agent
COPY plain ./plain
COPY tools ./tools
COPY adapters ./adapters
COPY obs ./obs
COPY service ./service
COPY eval ./eval
COPY mcp_server ./mcp_server
COPY mcp_client ./mcp_client
COPY src ./src
RUN uv venv --python 3.12 && \
    uv pip install --extra-index-url https://download.pytorch.org/whl/cpu torch && \
    uv pip install -e "/tau2-bench[knowledge]" faiss-cpu sentence-transformers "langgraph>=1.2.12,<2" "langgraph-checkpoint-sqlite>=3.0.0" "mcp>=2.0.0,<3" fastapi uvicorn pytest httpx
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
