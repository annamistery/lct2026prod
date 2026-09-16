# Production build guide

## Подготовка

- Ubuntu с NVIDIA driver и NVIDIA Container Toolkit.
- Docker Compose v2.
- Apache обслуживает `web/` по HTTPS и проксирует только `/api/` на `http://192.168.153.43:8030/api/`.
- Локальные DINO base/adapter и YOLO веса находятся в `models/`; runtime mount read-only.

## Сборка

```bash
cp .env.example .env
# заполнить пароль, CORS и при необходимости пути
docker compose build
docker compose up -d
docker compose ps
curl http://localhost:8030/api/ping
curl http://localhost:8030/api/ready
```

Сервис `migrate` выполняет `alembic upgrade head` после готовности PostgreSQL. API стартует только после успешной миграции.

## API

- `POST /api/products` — multipart `title`, `manufacturer`, `description`, `image`.
- `POST /api/v1/search` — полный кадр (`image`, `k`).
- `POST /api/v1/search-from-crop` — готовая этикетка (`image`, `k`).

Pipeline поиска: YOLO для полного кадра → DINOv2 embedding → exact pgvector по `product_embeddings` → лучший embedding каждого уникального товара → SIFT/RANSAC → Top-K. Один товар может иметь несколько эталонных изображений (`catalog`, `real`, `customer`). Изображения и кропы не поставляются через Git: таблицы `products`/`product_embeddings` и media заполняются из датасета заказчика.
