# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv/app
RUN apt-get update && apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
RUN python -m pip install --no-cache-dir . && groupadd --system app && useradd --system --gid app --home /srv/app app && mkdir -p /media /models && chown -R app:app /srv/app /media
USER app
EXPOSE 8030
HEALTHCHECK --interval=20s --timeout=5s --start-period=90s --retries=5 CMD curl --fail http://localhost:8030/api/ping || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8030", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "127.0.0.1"]
