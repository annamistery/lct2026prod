# LCT2026 Production

Демонстрационный сервис распознавания винных этикеток.

## Архитектура

- FastAPI: каталог и добавление товаров.
- PostgreSQL 16 + pgvector: `vector(384)` и exact cosine candidate search.
- DINOv2: глобальный embedding.
- SIFT/RANSAC: единственный финальный reranker.
- YOLO: серверный crop полного кадра; браузерный ONNX используется при возможности.

Search API версионируется: `/api/v1/search`, `/api/v1/search-from-crop`. Health, products и media не версионируются.

## Запуск

1. Установить Docker, NVIDIA Container Toolkit и Git LFS.
2. Выполнить `git lfs pull`.
3. Разместить локальную DINOv2 base model в `models/dinov2-small` и LoRA adapter/merged model в `models/dinov2_label_finetuned`.
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

Пошаговые команды для production-сервера: [`md/SERVER_COMMANDS.md`](md/SERVER_COMMANDS.md).

После каждой завершённой и локально проверенной доработки изменения фиксируются отдельным commit и сразу отправляются в `origin`.
