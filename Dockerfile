FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
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
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
