# LCT2026 Production

Демонстрационный сервис распознавания винных этикеток с AI-сомелье по каталогу.

## Архитектура

- **FastAPI**: каталог, добавление товаров, распознавание этикетки (v1, v2, v3, v4), диалог с AI-сомелье — один процесс, один порт `8030`.
- **PostgreSQL 16 + pgvector**: таблицы `products`, `product_embeddings` (DINOv2, vector 384) и `product_embeddings_v4` (SigLIP 2, vector 768), быстрый поиск ближайших векторов.
- **Поисковые пайплайны**:
  - **v1**: BBox YOLO detection + DINOv2-small (LoRA) cosine search (`/api/v1/search`, `/api/v1/search-from-crop`).
  - **v2**: Ректификация контура + DINOv2-small (`/api/v2/search`, `/api/v2/search-from-crop`).
  - **v3**: YOLO сегментация маски + DINOv2-base (768d) Letterbox 518×518 + SIFT (`/api/v3/search`).
  - **v4 (SOTA)**: YOLO сегментация + Google SigLIP 2 Vision Tower (768d, LoRA) Letterbox 518×518 + pgvector vote counting + OCR Vintage & Text Reranker (`/api/v4/search`, подробнее в [`docs/SEARCH_V4_API.md`](docs/SEARCH_V4_API.md)).
  - **Cascade (Финальный каскад)**: Двухэтапный гибридный поиск: быстрый v1 DINOv2 (~15 мс) на уникальных этикетках + адаптивный арбитраж соседей через v4 SigLIP 2/OCR (`/api/cascade/search`, `/api/cascade/predict`, подробнее в [`docs/SEARCH_CASCADE_API.md`](docs/SEARCH_CASCADE_API.md)).
- **AI-сомелье** (`app/sommelier`, см. [`app/sommelier/README.md`](app/sommelier/README.md)): детерминированный движок подбора вин к блюду по каталогу `wines_integrated.csv` — независимый от БД и GPU, собирается из CSV при старте API. `GET /api/sommelier/wine/{slug}` — точка интеграции с результатом распознавания этикетки.

Search API версионируется (`/api/v1/*`, `/api/v4/*`). Health, products, media и sommelier не версионируются.

---

## Запуск «из коробки» (Docker)

Проект полностью разворачивается и запускается в Docker на чистом сервере одной цепочкой команд:

### 1. Клонирование репозитория и загрузка моделей / данных через Git LFS

Для работы требуются Docker, NVIDIA Container Toolkit и Git LFS:
```bash
git clone https://github.com/vadfe/lct2026prod.git
cd lct2026prod

# Загрузка бинарных файлов (модели и дамп базы данных)
git lfs pull
sha256sum --check models/MODEL_MANIFEST.sha256
```

### 2. Конфигурация окружения

```bash
cp .env.example .env
sed -i "s/^APP_UID=.*/APP_UID=$(id -u)/; s/^APP_GID=.*/APP_GID=$(id -g)/" .env
```
*(При необходимости укажите свой пароль PostgreSQL в `.env`)*

### 3. Восстановление базы данных и медиа-каталога

В репозитории поставляется полный боевой дамп базы данных (товары + pgvector эмбеддинги) и эталонные медиа-кропы в папке `data/`. Для автоматического восстановления «из коробки» выполните:
```bash
chmod +x scripts/*.sh
./scripts/restore_production_data.sh
```
Скрипт автоматически:
1. Запустит контейнер базы данных PostgreSQL (`pgvector`).
2. Дождется готовности PostgreSQL.
3. Зальет полный SQL-дамп каталога и всех векторов `product_embeddings` / `product_embeddings_v4`.
4. Применит миграции Alembic (`alembic upgrade head`).
5. Распакует кропы эталонов в папку `./media`.
6. Перезапустит API и проверит доступность эндпоинта здоровья `/api/ping`.

### 4. Запуск всех сервисов

```bash
docker compose up -d --build
```

### 5. Проверка готовности системы

```bash
curl -fsS http://127.0.0.1:8030/api/ping
curl -fsS http://127.0.0.1:8030/api/ready
```
Ожидаемый ответ:
```json
{"ready":true,"database":true,"models":true,"sommelier":true}
```

---

## Веб-интерфейсы и Swagger

- **Интерактивный каскадный сканер (Cascade v1+v4):** `http://<server-ip>:8030/search-cascade` (полная диагностика всех шагов: v1 DINOv2, логика отбора соседей, v4 арбитраж SigLIP 2/OCR, Lightbox).
- **Мобильный сканер каскада (Cascade v1+v4):** `http://<server-ip>:8030/mobi-cascade` (встроенный) и автономная сборка в `export/mobitest_cascade.html`.
- **Интерактивный сканер v4 (веб):** `http://<server-ip>:8030/search-v4` (загрузка фото, визуализация BBox/маски, 4 карточки цепочки поиска и Lightbox).
- **Мобильный сканер с камеры:** `http://<server-ip>:8030/mobi/`
- **Swagger API документация:** `http://<server-ip>:8030/api/docs`

---

## Ручная загрузка моделей (если не используется Git LFS)

Все модели (DINOv2, SigLIP 2, YOLO) отслеживаются в Git через Git LFS. Если по какой-то причине Git LFS недоступен, базовые веса можно загрузить напрямую из Hugging Face:
- DINOv2-small: `facebook/dinov2-small` → в `models/dinov2-small/`
- SigLIP 2: `google/siglip2-base-patch16-512` → в `models/siglip2-base-patch16-512/`
Файлы весов LoRA-адаптеров (`models/dinov2_label_finetuned/` и `models/siglip2_v4_finetuned/`), а также детекторы YOLO (`models/yolo_label.pt`, `models/yolo_seg.pt`) поставляются в составе Git LFS репозитория.

---

## Проверки


```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
```

`pytest` покрывает и движок сомелье без БД/GPU (`tests/test_sommelier.py`, `tests/test_sommelier_service.py`); для диалога без HTTP есть `python scripts/sommelier_cli.py`.

Короткий первый запуск: [`md/START.md`](md/START.md). Расширенные серверные команды: [`md/SERVER_COMMANDS.md`](md/SERVER_COMMANDS.md).

После каждой завершённой и локально проверенной доработки изменения фиксируются отдельным commit и сразу отправляются в `origin`.
