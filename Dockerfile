# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
WORKDIR /srv/app

# Layer 1: System packages (cached)
RUN apt-get update && apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Layer 2: Heavy GPU wheels - downloaded and cached ONCE, never invalidated by app code changes
RUN python -m pip install --no-cache-dir --upgrade --extra-index-url https://download.pytorch.org/whl/cu128 "torch==2.9.0+cu128" "torchvision==0.24.0+cu128"

# Layer 3: Python dependencies from pyproject.toml (cached)
COPY pyproject.toml README.md ./
RUN python -m pip install --no-cache-dir .

# Layer 4: Application code and configs (fast layer)
COPY app ./app
COPY web ./web
COPY export ./export
COPY alembic ./alembic
COPY alembic.ini ./
RUN python -m pip install --no-cache-dir --no-deps -e . && \
    python -c "import app.main; import torch; print('PyTorch version:', torch.__version__, 'CUDA available:', torch.cuda.is_available(), 'Archs:', torch.cuda.get_arch_list())" && \
    groupadd --system app && useradd --system --gid app --home /srv/app app && mkdir -p /media /models && chown -R app:app /srv/app /media

USER app
EXPOSE 8030
HEALTHCHECK --interval=20s --timeout=5s --start-period=90s --retries=5 CMD curl --fail http://localhost:8030/api/ping || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8030", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "127.0.0.1"]
