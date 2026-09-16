# LCT2026 Production

Демонстрационный сервис распознавания винных этикеток.

## Архитектура

- FastAPI: каталог и добавление товаров.
- PostgreSQL 16 + pgvector: отдельные `products` и `product_embeddings`, несколько `vector(384)` на товар и exact cosine candidate search.
- DINOv2: глобальный embedding каждого эталонного изображения.
- SIFT/RANSAC: единственный финальный reranker.
- YOLO: серверный crop полного кадра; браузерный ONNX используется при возможности.

Search API версионируется: `/api/v1/search`, `/api/v1/search-from-crop`. Health, products и media не версионируются. pgvector выбирает ближайший embedding каждого уникального товара, после чего SIFT формирует финальный порядок. Изображения и кропы в Git не поставляются: PostgreSQL и media наполняются из датасета заказчика.

## Запуск

1. Установить Docker, NVIDIA Container Toolkit и Git LFS.
2. Выполнить `git lfs pull` — production YOLO, ONNX, базовая DINOv2 и финальный LoRA adapter поставляются из Git LFS.
3. Проверить модели: `sha256sum --check models/MODEL_MANIFEST.sha256`.
4. Скопировать `.env.example` в `.env` и заменить пароль БД и CORS origin.
5. Запустить `docker compose up --build`.
6. Проверить `http://localhost:8030/api/ping`, затем `/api/ready`.

Swagger: `/api/docs`. Web-каталог `web/` обслуживается внешним Apache по HTTPS.

## Проверки

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
```

Короткий первый запуск: [`md/START.md`](md/START.md). Расширенные серверные команды: [`md/SERVER_COMMANDS.md`](md/SERVER_COMMANDS.md).

После каждой завершённой и локально проверенной доработки изменения фиксируются отдельным commit и сразу отправляются в `origin`.
