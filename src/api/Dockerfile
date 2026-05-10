# syntax=docker/dockerfile:1.7
FROM python:3.11.15-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    PYTHONPATH=/app/src

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY src/api/requirements-core.txt /tmp/requirements-core.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r /tmp/requirements-core.txt

COPY src/api/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r /tmp/requirements.txt

COPY pyproject.toml README.md /app/
COPY src/gr_sat /app/src/gr_sat
COPY src/api /app/src/api

WORKDIR /app/src/api

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app/src/api", "--reload-dir", "/app/src/gr_sat"]
