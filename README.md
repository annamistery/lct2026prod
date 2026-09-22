# LCT2026 Production

Демонстрационный сервис распознавания винных этикеток с AI-сомелье по каталогу.

## Архитектура

- FastAPI: каталог, добавление товаров, распознавание этикетки, диалог с AI-сомелье — один процесс, один порт.
- PostgreSQL 16 + pgvector: отдельные `products` и `product_embeddings`, несколько `vector(384)` на товар и exact cosine candidate search.
- DINOv2: глобальный embedding каждого эталонного изображения.
- SIFT/RANSAC: единственный финальный reranker.
- YOLO: серверный crop полного кадра; браузерный ONNX используется при возможности.
- AI-сомелье (`app/sommelier`, см. [`app/sommelier/README.md`](app/sommelier/README.md)): детерминированный движок подбора вин к блюду по каталогу `wines_integrated.csv` — независимый от БД и GPU, собирается из CSV при старте API. `GET /api/sommelier/wine/{slug}` — точка интеграции с результатом распознавания этикетки (slug товара из поиска → справка сомелье и, если передать блюдо, оценка сочетания).

Search API версионируется: `/api/v1/search`, `/api/v1/search-from-crop`. Health, products, media и sommelier не версионируются. pgvector выбирает ближайший embedding каждого уникального товара, после чего SIFT формирует финальный порядок. Изображения и кропы в Git не поставляются: PostgreSQL и media наполняются из датасета заказчика.

## Запуск

Интегрированный пайплайн — один `docker compose up`: поднимает БД, применяет миграции, затем в одном API-контейнере параллельно загружает ML-модели (YOLO/DINOv2) и каталог сомелье (CSV → в памяти), после чего обслуживает и распознавание, и диалог с сомелье на одном порту.

1. Установить Docker, NVIDIA Container Toolkit и Git LFS.
2. Выполнить `git lfs pull` — production YOLO, ONNX, базовая DINOv2 и финальный LoRA adapter поставляются из Git LFS.
3. Проверить модели: `sha256sum --check models/MODEL_MANIFEST.sha256`.
4. Скопировать `.env.example` в `.env` и заменить пароль БД и CORS origin.
5. Запустить `docker compose up --build`.
6. Проверить `http://localhost:8030/api/ping`, затем `/api/ready` — в ответе `sommelier: true` означает, что каталог сомелье (2103 вина) собрался и `/api/sommelier/*` готов; на итоговый флаг `ready` это поле не влияет (сомелье не зависит от БД/GPU).
7. Проверить сомелье:
   ```bash
   curl -fsS -X POST http://localhost:8030/api/sommelier/ask -H 'Content-Type: application/json' -d '{"message": "стейк рибай, не люблю дуб"}'
   ```

Swagger: `/api/docs`. Web-каталог `web/` обслуживается внешним Apache по HTTPS. Одиночная загрузка выполняется через `POST /api/products`, пакетная JSON/CSV загрузка — через фоновый `POST /api/imports`; формат описан в [`docs/BATCH_IMPORT.md`](docs/BATCH_IMPORT.md). Диалог, справка по вину и конфигурация сомелье — в [`docs/SOMMELIER.md`](docs/SOMMELIER.md).

## Проверки

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
```

`pytest` покрывает и движок сомелье без БД/GPU (`tests/test_sommelier.py`, `tests/test_sommelier_service.py`); для диалога без HTTP есть `python scripts/sommelier_cli.py`.

Короткий первый запуск: [`md/START.md`](md/START.md). Расширенные серверные команды: [`md/SERVER_COMMANDS.md`](md/SERVER_COMMANDS.md).

После каждой завершённой и локально проверенной доработки изменения фиксируются отдельным commit и сразу отправляются в `origin`.
