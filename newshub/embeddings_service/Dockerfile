# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12-slim

FROM python:${PYTHON_VERSION} AS builder
WORKDIR /build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev git \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry poetry-plugin-export

COPY pyproject.toml poetry.lock* /build/

RUN poetry export --with embeddings --without-hashes -f requirements.txt -o /build/requirements.txt \
 && pip wheel --wheel-dir /build/wheels -r /build/requirements.txt


FROM python:${PYTHON_VERSION} AS runtime
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq5 \
  && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/wheels /wheels
COPY --from=builder /build/requirements.txt /requirements.txt

RUN pip install --no-cache-dir --no-index --find-links=/wheels -r /requirements.txt \
 && rm -rf /wheels /requirements.txt

COPY . /app

WORKDIR /app/embeddings_service

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9001"]
