# ============================================================================
# aegis-mem Production Dockerfile (Python 3.12 Slim, Non-Root User)
# ============================================================================
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Create non-root system user and data directory for SQLite telemetry
RUN groupadd --gid 10001 aegis && \
    useradd --uid 10001 --gid aegis --shell /bin/bash --create-home aegis && \
    mkdir -p /app/data && chown -R aegis:aegis /app

# Copy project metadata and source code
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install package and production dependencies
RUN pip install --no-cache-dir . && chown -R aegis:aegis /app

USER aegis

EXPOSE 8000

ENTRYPOINT ["uvicorn", "aegis_mem.interfaces.cloud_api:app", "--host", "0.0.0.0", "--port", "8000"]

