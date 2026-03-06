FROM python:3.11-slim

WORKDIR /app

# System deps for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
# Install server extras (fastapi, uvicorn, sse-starlette, etc.)
RUN pip install --no-cache-dir -e ".[server]" 2>/dev/null || pip install --no-cache-dir -e "."

COPY . .
# Re-install in editable mode now that all source is present
RUN pip install --no-cache-dir -e ".[server]" 2>/dev/null || pip install --no-cache-dir -e "."

# Default: run the API server
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
