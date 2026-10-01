# Архитектура LCT2026 Production

Сервис распознаёт винные этикетки по фотографии и находит вино в каталоге. В проект также входит AI-сомелье, который подбирает вина к блюду. Всё работает в одном процессе FastAPI на порту `8030`. Товары и векторы изображений хранятся в PostgreSQL 16 с расширением pgvector.

```
Клиент (браузер / мобильный сканер / participant_test.sh)
        │  HTTP :8030  (в проде Apache проксирует только /api/)
        ▼
FastAPI (app/main.py)
  ├─ /api/*            JSON API: health, products, imports, search v1–v4, cascade, twins, sommelier
  ├─ /, /search-*, …   HTML-страницы (Jinja-шаблоны app/web/templates)
  ├─ /mobi/            мобильный клиент (web/)
  └─ /export/          автономный мобильный сканер каскада (export/)
        │
        ├─ ML (GPU): YOLO detect/seg → DINOv2-small LoRA (384d) → SigLIP 2 LoRA (768d) → слияние оценок и зоны ответа
        ├─ PostgreSQL + pgvector: products, product_embeddings, product_embeddings_v2/v3/v4, import_jobs
        └─ Файлы: /models (read-only), /media (кропы и медиа товаров), /imports (входные данные каталога)
```

---

## 1. Дерево проекта

Большие бинарные файлы (веса моделей, дампы БД, архив медиа) хранятся в Git LFS. Содержимое `media/`, `imports/inbox`, `imports/staging` и `.env` в Git не попадает.

```
lct2026prod/
├── AGENTS.md                  # правила разработки и деплоя (обязательны для всех изменений)
├── README.md                  # обзор и быстрый запуск
├── Architecture.md            # этот документ
├── Dockerfile                 # образ API: PyTorch 2.x + CUDA, uvicorn на :8030
├── compose.yaml               # сервисы db (pgvector), migrate (alembic), api (GPU)
├── pyproject.toml             # зависимости, настройки pytest / ruff / mypy
├── alembic.ini                # конфигурация миграций
├── .env.example               # шаблон переменных окружения
├── .gitattributes             # правила Git LFS (*.pt, *.bin, *.safetensors, data/*.gz …)
├── .github/workflows/ci.yml   # CI: миграции, alembic check, ruff, pytest, docker build
│
├── app/                       # backend-приложение (Python-пакет)
│   ├── main.py                # создание FastAPI, middleware X-Request-ID, CORS, static-mount
│   ├── api/                   # HTTP-роуты по функциональным разделам
│   │   ├── router.py          # общий роутер с префиксом /api
│   │   ├── health/            # /api/ping, /api/ready
│   │   ├── products/          # /api/products, /api/media/{path}
│   │   ├── imports/           # /api/imports — фоновые задачи пакетного импорта
│   │   ├── search/
│   │   │   ├── v1/            # /api/v1/search, search-from-crop, eval/predict
│   │   │   ├── v2/            # /api/v2/* (ректификация контура)
│   │   │   ├── v3/            # /api/v3/* (DINOv2-base + SIFT)
│   │   │   ├── v4/            # /api/v4/* (SigLIP 2 + OCR)
│   │   │   └── cascade/       # /api/cascade/search, search-from-crop, predict
│   │   ├── twins/             # /api/twins/clusters — кластеры визуально похожих вин
│   │   └── sommelier/         # /api/sommelier/ask, reset, wine/{slug}
│   ├── core/                  # config.py (Settings), lifespan.py (загрузка моделей),
│   │                          # dependencies.py (DI), logging.py
│   ├── db/                    # base.py, session.py, models/ (ORM), repositories/ (запросы pgvector)
│   ├── pipelines/search/      # бизнес-логика поиска: v1, v2, v3, v4, cascade
│   ├── schemas/               # Pydantic-схемы запросов и ответов
│   ├── services/              # ML- и инфраструктурные сервисы (детектор, эмбеддинги, OCR …)
│   ├── sommelier/             # движок AI-сомелье + data/wines_integrated.csv
│   └── web/                   # HTML-роуты, тестовые стенды detect/rectify, templates/, static/
│
├── alembic/versions/          # миграции схемы (0001 products … 0007 embeddings_v4)
├── web/                       # мобильный клиент камеры (/mobi/): index.html, app.js, detector.js
├── export/                    # автономный мобильный сканер каскада (ONNX YOLO в браузере)
├── scripts/                   # импорт каталога, построение эмбеддингов, обучение, бенчмарки, бэкап
├── train/losses.py            # SupCon loss для дообучения
├── tests/                     # pytest
├── models/                    # веса моделей (Git LFS) + MODEL_MANIFEST.sha256
├── data/                      # боевые дампы БД и архив медиа (Git LFS)
├── imports/                   # входные данные: inbox/ (архивы), staging/<batch_id>/, images/
├── media/                     # записываемое хранилище медиа товаров (не в Git)
├── tmp/1, tmp/2               # тестовые наборы для оценки (mapping.json, eval.sh)
├── docs/                      # эксплуатационная и API-документация
└── md/                        # инструкции по запуску, журнал разработки, roadmap, participant_test.sh
```

---

## 2. Краткое описание модулей

### 2.1 `app/core` — ядро

| Модуль | Назначение |
|---|---|
| `config.py` | `Settings` на pydantic-settings: читает `.env` и переменные окружения. Хранит пути к моделям и медиа, пороги, лимиты загрузки и параметры каскада. Методы `resolved_*_path` находят модели в `/models`, `media/models` или `./models`. |
| `lifespan.py` | При старте создаёт engine БД и загружает сомелье, YOLO (детекция и сегментация), ректификацию и пайплайны v1–v4. Затем собирает каскад. Каждый пайплайн кладётся в `app.state`. Если модели для v3/v4 нет, пайплайн пропускается, и API всё равно стартует. |
| `dependencies.py` | FastAPI-зависимости: сессия БД, сервисы и пайплайны из `app.state`. Если нужный компонент не загружен, возвращает 503. |
| `logging.py` | Настройка формата и уровня логов. |

### 2.2 `app/db` — база данных

| Модуль | Назначение |
|---|---|
| `base.py`, `session.py` | Declarative `Base`, async engine и фабрика сессий (asyncpg). |
| `models/product.py` | `Product`: slug, название, производитель, описание, пути к фото и кропу. `ProductEmbedding` (v1, 384d), `ProductEmbeddingV2` (384d), `ProductEmbeddingV3` (768d) и `ProductEmbeddingV4` (768d, SigLIP 2) — у одного товара может быть много векторов (`catalog`, `augmented`, `real`, `customer`). |
| `models/import_job.py` | Надёжные задачи пакетного импорта и их элементы. |
| `repositories/products.py` | `ProductRepository`: список и поиск товаров, `nearest*` — поиск ближайших векторов в pgvector по косинусной мере. Выбирает лучший вектор для каждого уникального товара. |

### 2.3 `app/services` — сервисы

| Модуль | Назначение |
|---|---|
| `images.py` | Декодирует загрузки с проверкой размера и числа пикселей, делает crop и canonical resize. Атомарно сохраняет `source.webp`/`label.webp` в `media/products/<uuid>/`. |
| `detector.py` | YOLO bbox-детектор этикетки (`models/yolo_label.pt`). |
| `segmenter.py` | YOLO-сегментация: полигон маски этикетки (`models/yolo_seg.pt`). |
| `rectification.py` | Выпрямление перспективы и «развёртка» цилиндрической этикетки (используется в v2). |
| `query_prep_v3.py` | Подготовка запроса: сегментация, кроп, letterbox 518×518 (`letterbox_pil`). Используется в v1, v3, v4 и каскаде. |
| `embeddings.py` | DINOv2 + LoRA (PEFT): эмбеддинги одного изображения или батча. |
| `siglip_embeddings.py` | Vision Tower SigLIP 2 + LoRA: эмбеддинги 768d для v4. |
| `ocr_reranker.py` | Второй этап v4: OCR извлекает год урожая и текст, итоговый скор взвешивает сходство, винтаж и текст. |
| `augment.py` | «Камерные» аугментации: 23 типа × 5 вариантов + оригинал = 116 изображений на товар. |
| `product_ingestion.py` | Создание товара из фото: детекция, кроп, эмбеддинг, запись в БД и медиа. |
| `batch_import.py` | Фоновый импорт из `imports/staging/<batch_id>/<manifest>`. Восстанавливает прерванные задачи после рестарта. |
| `sommelier_service.py` | Обёртка над движком сомелье: сессии диалогов с TTL и лимитом, запись фидбека в `media/sommelier/`. |

### 2.4 `app/pipelines/search` — поисковые пайплайны

| Пайплайн | Схема |
|---|---|
| **v1** | YOLO bbox → кроп → DINOv2-small LoRA (384d) → pgvector top-K (`reranking.py` — опциональный SIFT). |
| **v2** | Ректификация контура → DINOv2-small → `product_embeddings_v2`. |
| **v3** | YOLO-сегментация → letterbox 518 → DINOv2-base 768d → подсчёт голосов → SIFT-реранк по аугментированным кропам. Загружается, только если есть модель. |
| **v4** | YOLO-сегментация → letterbox 518 → SigLIP 2 LoRA 768d → подсчёт голосов по пулу соседей → OCR-реранкер (винтаж и текст). |
| **cascade** | Основной продовый режим. Два вида кропа (letterbox и маска сегментации); DINOv2 отбирает 30 кандидатов, для них считается оценка SigLIP 2 (оба вида) + 0,3 · DINOv2 (оба вида). `decision.py` по сходству SigLIP 2 и отрыву от второго кандидата выбирает зону ответа: `found` / `probable` / `not_in_catalog` («Данного вина нет в каталоге», `slug = null`). OCR не используется. Если SigLIP 2 недоступен, каскад работает на DINOv2 со статусом `probable`. Точность и калибровка — `docs/RECOGNITION_QUALITY.md`. |

### 2.5 `app/api` — HTTP API

Все внешние эндпоинты начинаются с `/api/`. Версионируется только поиск.

| Раздел | Эндпоинты |
|---|---|
| health | `GET /api/ping` (процесс жив), `GET /api/ready` (БД + pgvector + YOLO/DINO; при неготовности 503) |
| products | `GET/POST /api/products`, `GET /api/products/{id}`, `GET /api/media/{path}` |
| imports | `POST /api/imports` (batch_id + manifest → 202), `GET /api/imports/{job_id}` |
| search v1–v4 | `POST /api/vN/search`, `/api/vN/search-from-crop`, `/api/vN/eval/predict` |
| cascade | `POST /api/cascade/search`, `/search-from-crop`, `/predict` — в ответе `status` (`found` / `probable` / `not_in_catalog`); для «нет в каталоге» `slug = null` |
| twins | `GET /api/twins/clusters` — группы похожих этикеток (union-find по близости векторов) |
| sommelier | `POST /api/sommelier/ask`, `POST /api/sommelier/reset`, `GET /api/sommelier/wine/{slug}?dish=`, `GET /api/sommelier/wine/{slug}/alternatives` — похожие по стилю вина к распознанному |

Подробнее: `docs/SEARCH_CASCADE_API.md`, `docs/SEARCH_V4_API.md`, `docs/SOMMELIER.md`, Swagger `/api/docs`.

### 2.6 `app/web` — веб-интерфейс

`router.py` отдаёт HTML-страницы:

- `/` — редирект на главную;
- `/products` и `/products/{id}` — каталог;
- `/add` — добавление товара;
- `/import` — пакетный импорт;
- `/search`, `/search-v2`, `/search-v3`, `/search-v4`, `/search-cascade` — сканеры с диагностикой;
- `/mobi-cascade` — мобильный сканер каскада;
- `/twins` — похожие этикетки.

`detect_router.py` (`/detect-test`) и `rectify_router.py` (`/rectify-test`) — отладочные стенды детекции и ректификации.

### 2.7 `app/sommelier` — AI-сомелье

Работает детерминированно, без БД и GPU. Каталог собирается из `data/wines_integrated.csv` при старте API.

- `catalog.py` — загрузка CSV, сборка каталога;
- `features.py` — признаки вина только из полей базы, с цитатами-доказательствами;
- `dishes.py` — профиль блюда по вкусовым осям;
- `context.py` — разбор свободного текста;
- `engine.py` — фильтры, скоринг и диверсификация;
- `explain.py` — объяснения для гостя;
- `guardrails.py` — этические ограничения;
- `sommelier.py` — оркестратор диалога;
- `llm.py` — необязательный LLM-слой: нужны пакет `anthropic` и `ANTHROPIC_API_KEY`, без них ответы строятся по шаблонам.

### 2.8 Клиенты

- `web/` — мобильный клиент камеры (`/mobi/`), настройки в `config.js`.
- `export/` — автономный мобильный сканер каскада: `mobitest_cascade.html`, YOLO в браузере через ONNX Runtime (`models/yolov8_label.onnx`, `vendor/ort.all.min.js`). Адрес backend задаётся в `export/config.js` (`MOBI_API_BASE`).

### 2.9 `scripts/` — утилиты

| Группа | Скрипты |
|---|---|
| Каталог | `ingest_cascade_catalog.py` (**основной**: CSV + фото → v1 и v4 эмбеддинги, по 116 на товар), `import_from_csv.py`, `download_images.py`, `list_missing_catalog_images.py`, `prepare_dataset_import.sh`, `clear_database.py` |
| Эмбеддинги (legacy) | `build_embeddings_v2.py`, `build_embeddings_v3.py`, `build_embeddings_v4.py` |
| Обучение | `build_dataset_v3.py`, `build_dataset_v4.py`, `train_dino_v3.py`, `train_siglip_v4.py` (+ `train/losses.py`) |
| Оценка | `eval_recognition.py` (**основной**: точность и зоны ответа через API по `mapping.json` / `labels.tsv`), `research_dump_features.py` + `calibrate_cascade.py` (калибровка порогов без GPU), `audit_catalog.py` (дубли этикеток в каталоге), `run_benchmark.py`, `eval_results.py`, `eval_perspective_frame.py`, `md/participant_test.sh` |
| Данные и модели | `restore_production_data.sh`, `export_production_data.sh`, `stage_production_models.sh`, `collect_model_inventory.sh` |
| Сомелье | `sommelier_cli.py` — диалог в консоли без HTTP |

### 2.10 Модели (`models/`)

| Путь | Назначение |
|---|---|
| `yolo_label.pt` | Детектор bbox этикетки |
| `yolo_seg.pt` | Сегментация этикетки |
| `dinov2-small/` + `dinov2_label_finetuned/` | База и LoRA-адаптер v1 (384d) |
| `siglip2-base-patch16-512/` + `siglip2_v4_finetuned/` | База и LoRA-адаптер v4 (768d) |
| `MODEL_MANIFEST.sha256` | Контрольные суммы весов |

Во время работы модели только читаются, в контейнере они смонтированы как `/models:ro`.

---

## 3. Настройка

### 3.1 Требования

- **Сервер (прод):** Ubuntu, Docker + Docker Compose, NVIDIA GPU + NVIDIA Container Toolkit, Git LFS.
- **Локальная разработка:** Python 3.11–3.12, PostgreSQL 16 с pgvector (можно поднять только сервис `db` из `compose.yaml`).

### 3.2 Переменные окружения (`.env`)

Скопируйте шаблон: `cp .env.example .env`. Основные параметры:

| Переменная | Назначение |
|---|---|
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | Учётные данные контейнера БД |
| `DATABASE_URL` | `postgresql+asyncpg://<user>:<pass>@db:5432/<db>`. Пароль должен совпадать с `POSTGRES_PASSWORD`. Локально вместо `db` укажите `localhost`. |
| `APP_UID`, `APP_GID` | UID/GID владельца host-каталога `media/`: `id -u` / `id -g` |
| `MEDIA_DIR`, `IMPORT_STAGING_DIR` | Пути к медиа и staging импорта (в контейнере `/media`, `/imports/staging`) |
| `DINO_MODEL_PATH`, `DINO_BASE_MODEL_PATH`, `YOLO_MODEL_PATH` | Пути к моделям v1 (в контейнере `/models/...`) |
| `SIGLIP_V4_MODEL_PATH`, `SIGLIP_V4_BASE_MODEL_PATH`, `YOLO_SEG_MODEL_PATH` | Пути к моделям v4 и сегментации. Если не заданы, ищутся автоматически. |
| `CORS_ORIGINS` | Разрешённые origin через запятую. Пусто, если web и `/api/` отдаются с одного адреса. |
| `MAX_UPLOAD_BYTES`, `MAX_IMAGE_PIXELS`, `MAX_TOP_K` | Лимиты входных изображений и выдачи |
| `CANDIDATE_POOL_SIZE`, `GPU_CONCURRENCY`, `SIFT_CONCURRENCY` | Производительность. При GPU OOM уменьшите `GPU_CONCURRENCY`. |
| `CASCADE_CANDIDATE_POOL`, `CASCADE_V1_WEIGHT`, `CASCADE_FOUND_MIN_SIMILARITY`, `CASCADE_FOUND_MIN_MARGIN`, `CASCADE_REJECT_BELOW_SIMILARITY`, `CASCADE_PREDICT_THRESHOLD` | Каскад: число кандидатов, вес DINOv2, пороги зон «найдено» и «нет в каталоге» (`docs/SEARCH_CASCADE_API.md`) |
| `ENABLE_OCR_RERANK_V4`, `OCR_RERANK_WEIGHT_*` | OCR-реранкер отдельного пайплайна v4 (каскад OCR не использует) |
| `SOMMELIER_MAX_SESSIONS`, `SOMMELIER_SESSION_TTL_SECONDS` | Сессии сомелье |
| `LOG_LEVEL` | Уровень логов |

Полный список с значениями по умолчанию — `app/core/config.py`. Имя переменной — это имя поля `Settings` в верхнем регистре.

### 3.3 Изменение схемы БД

Схему меняйте только через SQLAlchemy-модели и новую ревизию Alembic. `create_all` в рантайме запрещён.

```bash
alembic revision --autogenerate -m "описание"
alembic upgrade head
alembic check        # CI проверяет, что модели и миграции совпадают
```

---

## 4. Запуск

### 4.1 Прод / сервер (Docker)

```bash
git clone https://github.com/vadfe/lct2026prod.git
cd lct2026prod
git lfs pull
sha256sum --check models/MODEL_MANIFEST.sha256

cp .env.example .env
sed -i "s/^APP_UID=.*/APP_UID=$(id -u)/; s/^APP_GID=.*/APP_GID=$(id -g)/" .env
nano .env                                  # задать POSTGRES_PASSWORD и тот же пароль в DATABASE_URL

chmod +x scripts/*.sh
./scripts/restore_production_data.sh       # БД (products + v1/v4 векторы) и media/ из data/*.gz

docker compose up -d --build               # db → migrate (alembic upgrade head) → api
```

Проверка:

```bash
curl -fsS http://127.0.0.1:8030/api/ping
curl -fsS http://127.0.0.1:8030/api/ready
# {"ready":true,"database":true,"models":true,"sommelier":true}
```

Веб-интерфейсы:

- `http://<host>:8030/search-cascade` — основной сканер;
- `http://<host>:8030/mobi-cascade` и `http://<host>:8030/mobi/` — мобильные сканеры;
- `http://<host>:8030/products` — каталог;
- `http://<host>:8030/api/docs` — Swagger.

В проде Apache проксирует наружу только `/api/`.

### 4.2 Локальный запуск без Docker

```bash
python -m venv .venv
.venv\Scripts\activate                     # Windows (Linux/macOS: source .venv/bin/activate)
python -m pip install -e ".[dev]"

docker compose up -d db                    # или свой PostgreSQL 16 + pgvector
# в .env: DATABASE_URL=postgresql+asyncpg://lct:<pass>@localhost:5432/lct2026
#         MEDIA_DIR=media, пути моделей — models/... (или удалить строки: сработают значения по умолчанию)

alembic upgrade head
uvicorn app.main:app --host 127.0.0.1 --port 8030 --reload
```

Откройте `http://localhost:8030/`. Без GPU модели работают на CPU, поиск будет медленнее. Если БД пустая, загрузите каталог (раздел 4.3) или восстановите дамп.

### 4.3 Загрузка каталога

1. Положите CSV в `./imports/wines_integrated_cleared.csv`, фото бутылок — в `./media/catalog_sources/`.
2. Выполните импорт:

```bash
# проверка без записи
docker compose exec api python scripts/ingest_cascade_catalog.py \
    --csv /imports/wines_integrated_cleared.csv --images-dir /media/catalog_sources --dry-run
# полная заливка (идемпотентна; --limit N — пробный прогон, --force-rebuild — перезапись)
docker compose exec api python scripts/ingest_cascade_catalog.py \
    --csv /imports/wines_integrated_cleared.csv --images-dir /media/catalog_sources
```

Большие архивы можно импортировать через API. Сначала распакуйте архив в staging:

```bash
./scripts/prepare_dataset_import.sh <archive.zip|tar.gz> <batch_id>
```

Затем вызовите `POST /api/imports` с `batch_id` и именем манифеста (подробнее — `docs/BATCH_IMPORT.md`).

### 4.4 Эксплуатация

```bash
docker compose ps
docker compose logs -f api
docker compose run --rm migrate alembic current
./scripts/export_production_data.sh        # бэкап БД + media в data/
```

БД и `media/` резервируются вместе: если восстановить только одно из них, ссылки на изображения будут битыми. Подробнее — `docs/RUNBOOK.md`, `md/SERVER_COMMANDS.md`.

---

## 5. Тестирование

### 5.1 Статические проверки и unit-тесты (локально)

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
```

| Тест | Что проверяет |
|---|---|
| `test_architecture.py` | Все роуты под `/api`, версионируется только поиск |
| `test_search_cascade.py` | Зоны ответа, ранжирование слиянием, «нет в каталоге», режим без SigLIP 2, прогрев |
| `test_search_v4.py` | Пайплайн v4 и OCR-реранкер |
| `test_reranking.py`, `test_rectification.py`, `test_images.py` | SIFT, ректификация, валидация изображений |
| `test_batch_import.py`, `test_repository.py`, `test_models.py` | Импорт, репозиторий, ORM-модели |
| `test_sommelier.py`, `test_sommelier_service.py` | Движок, альтернативы и сервис сомелье (без БД и GPU) |

Проверка миграций на живой БД (как в CI):

```bash
alembic upgrade head
alembic check
```

CI (`.github/workflows/ci.yml`) на каждый push запускает PostgreSQL pgvector, затем `alembic upgrade head`, `alembic check`, `ruff`, `pytest` и `docker build`.

### 5.2 Оценка качества распознавания (нужен запущенный сервер с каталогом)

```bash
# точность и зоны ответа каскада (найдено / похоже / нет в каталоге) по разметке
python scripts/eval_recognition.py --api http://127.0.0.1:8030 --sets tmp1=tmp/1 tmp2=tmp/2

# пересчёт порогов после изменения каталога (см. docs/RECOGNITION_QUALITY.md)
docker compose exec api python scripts/research_dump_features.py --sets tmp1=/srv/app/tmp/1 tmp2=/srv/app/tmp/2
python scripts/calibrate_cascade.py media/research

# прогон контрольных изображений через HTTP (формат участника)
bash md/participant_test.sh --images-dir queries --manifest queries.tsv \
    --endpoint http://127.0.0.1:8030/api/cascade/predict --output predictions.jsonl

# сравнение пайплайнов: v1, cascade/eval/predict, cascade/predict, v2, v3 (--skip-v2 / --skip-v3)
python scripts/run_benchmark.py --host http://127.0.0.1:8030
```

Подробная процедура — `md/ИНСТРУКЦИЯ_ПРОВЕРКИ.md`.

### 5.3 Ручная проверка

- `curl http://127.0.0.1:8030/api/ready` должен вернуть 200 и `"ready": true`.
- На `/search-cascade` загрузите фото этикетки: страница покажет все шаги каскада.
- Для сомелье выполните `python scripts/sommelier_cli.py "стейк рибай, не люблю дуб"` или откройте `POST /api/sommelier/ask` в Swagger.

---

## 6. Ключевые правила (из `AGENTS.md`)

- Все внешние эндпоинты начинаются с `/api/`. Версионируется только поиск.
- PostgreSQL + pgvector — единственное хранилище товаров и векторов. FAISS не используется.
- Схема меняется только через модели и Alembic.
- Пути берутся из конфигурации, модели во время работы только читаются, медиа записывается только в `MEDIA_DIR`.
- Входные значения запросов никогда не подставляются в SQL, shell или пути: используются bound-выражения и валидированные типы.
- В Git не попадают секреты, `.env`, данные PostgreSQL, загруженные медиа и кеши. Веса моделей разрешены только через Git LFS.
