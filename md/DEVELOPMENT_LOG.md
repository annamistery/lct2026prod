# Журнал разработки LCT2026 Production

## Сессия 2026-09-16

### Цель

Собрать с нуля чистый воспроизводимый production-проект для демонстрации заказчику, используя проверенные алгоритмы старого LCT2026 как источник, но не перенося legacy-архитектуру.

### Принятые архитектурные решения

- Backend: FastAPI, порт `8030`.
- Все внешние backend-маршруты находятся под `/api/`, поскольку Apache проксирует только этот prefix.
- Версионируются только поисковые pipeline: текущая версия — `/api/v1/*`.
- Общие health, products, imports, media, core, ORM и ML services не версионируются.
- Хранилище каталога и векторов: PostgreSQL 16 + pgvector.
- `products` хранит текстовую карточку и ссылки на файлы.
- `product_embeddings` хранит несколько эталонных изображений и `vector(384)` на один товар.
- Первичный поиск: exact cosine pgvector без ANN.
- Финальный и единственный reranker: SIFT + Lowe ratio + homography RANSAC.
- Полный кадр обрабатывается серверной YOLO; готовый клиентский crop минует YOLO.
- Production-модели полностью поставляются через Git LFS; runtime не зависит от Hugging Face cache, старого проекта или сетевой загрузки.
- Изображения товаров, кропы и датасеты в Git не хранятся. Они поступают из датасета заказчика в PostgreSQL и `media/`.
- Deployment: Docker Compose с GPU API, PostgreSQL/pgvector и отдельным Alembic migration service.

### Реализовано

#### API

- `GET /api/ping`
- `GET /api/ready`
- `GET /api/products`
- `GET /api/products/{id}`
- `POST /api/products` — одиночное добавление товара
- `GET /api/media/{path}`
- `POST /api/v1/search`
- `POST /api/v1/search-from-crop`
- `POST /api/imports` — запуск фонового пакетного импорта
- `GET /api/imports/{job_id}` — прогресс и ошибки импорта

#### ML pipeline

```text
полный кадр → YOLO crop ┐
готовый crop ───────────┴→ DINOv2 global embedding
                          → exact pgvector candidates
                          → лучший embedding каждого уникального товара
                          → SIFT/RANSAC reranking
                          → финальный Top-K и winner
```

DINO patch reranking намеренно отсутствует.

#### Модели в Git LFS

- `models/yolo_label.pt`
- `web/models/yolov8_label.onnx`
- `models/dinov2-small/model.safetensors`
- `models/dinov2_label_finetuned/adapter_model.bin` — финальный `dino_aug_116`
- конфигурации processor/base/LoRA и `models/MODEL_MANIFEST.sha256`

Все файлы моделей проверены через SHA-256 после чистого clone.

#### База данных

Baseline `20260916_0001` сразу создаёт финальные:

- extension `vector`;
- `products`;
- `product_embeddings`.

Migration `20260916_0002` добавляет durable batch import:

- `import_jobs`;
- `import_items`.

Каждый успешный импортированный товар получает минимум один `product_embeddings` с типом `catalog`. В дальнейшем поддерживаются типы `catalog`, `real`, `customer`.

#### Пакетный импорт датасета

- Поддерживаются `manifest.json` и `manifest.csv`.
- Изображения загружаются на сервер архивом ZIP или TAR/TAR.GZ.
- Архив кладётся в `imports/inbox/`.
- `scripts/prepare_dataset_import.sh` безопасно распаковывает его в `imports/staging/<batch_id>/`.
- Блокируются absolute paths, `..`, symlink, hardlink и device entries.
- API принимает только валидированный `batch_id` и имя manifest, а не произвольный серверный путь.
- Импорт выполняется фоном, прогресс и ошибки сохраняются в PostgreSQL.
- Одновременно разрешён один batch import.
- Прерванные перезапуском jobs помечаются failed.

Формат описан в `docs/BATCH_IMPORT.md`.

#### Безопасность и эксплуатация

- Request-scoped SQLAlchemy sessions с rollback/cleanup.
- SQLAlchemy bind expressions вместо SQL-интерполяции.
- Проверка формата, размера и количества пикселей изображений.
- Path confinement для media и staging.
- Ограничения Top-K, candidate pool, GPU и SIFT concurrency.
- Non-root API container запускается с UID/GID владельца host media bind mount.
- Модели монтируются read-only.
- JSON logging и request ID.
- Liveness отделён от readiness.
- Все завершённые изменения фиксируются commit и отправляются в `origin/main`.

### Проверки в этой сессии

Локально выполнены:

- Python AST parsing всех новых Python-файлов;
- Ruff — `All checks passed` после добавления batch import;
- Bash syntax checks для server scripts;
- JavaScript syntax checks;
- SHA-256 сравнение исходных и перенесённых YOLO/ONNX;
- Git LFS attribute/status checks;
- проверка отсутствия FAISS и legacy imports в новом runtime.

На Ubuntu GPU-сервере подтверждено:

- RTX 5070 Ti видна хостом и CUDA 12.4 container runtime;
- Docker и Compose работают;
- чистый clone получает все четыре model artifacts через Git LFS;
- checksums полного model bundle проходят;
- PostgreSQL/pgvector запускается и baseline Alembic migration применяется;
- после исправлений `/api/ready` возвращал HTTP 200:

```json
{"ready": true, "database": true, "models": true}
```

Последняя версия с batch import (`d845b08`) запушена, но её migration и endpoints ещё должны быть проверены на сервере после `git pull` и `docker compose up -d --build`.

### Исправленные проблемы развёртывания

1. Несовместимость `numpy==2.1.3` с `ultralytics==8.3.75` — закреплён `numpy==2.1.1`.
2. `ProductRepository.list` затенял встроенный `list` в аннотации — метод переименован в `list_products`, добавлены deferred annotations и Docker import smoke check.
3. API не мог писать в bind mount `/media` — Compose запускает API с `APP_UID`/`APP_GID` владельца checkout.
4. Предварительная история Alembic была схлопнута в один финальный baseline до первого deployment.
5. Скрипт staging моделей теперь проверяет Git identity, поддерживает безопасное продолжение и повторную checksum-проверку.

### История ключевых commits

- `f6ddd7f` — базовый production-проект.
- `c8a0d99` — серверная инструкция перенесена в `md/`.
- `dd112ef` — bootstrap Git LFS.
- `0b737a5` — инвентаризация legacy ML-артефактов.
- `135422e` — подготовка воспроизводимого model bundle.
- `f4588ee` — полный комплект моделей добавлен в Git LFS.
- `901b765` — безопасный повторный запуск model staging.
- `a4d5395` — разделены products и product embeddings.
- `1b88774` — чистый baseline Alembic и короткий START.
- `0e2478a` — совместимая версия NumPy.
- `a834ab1` — устранено затенение аннотаций, Docker import smoke check.
- `499415b` — исправлены права media bind mount через UID/GID.
- `d845b08` — durable JSON/CSV batch import.

## Сессия 2026-09-18

### Цель и контекст
Подготовка к наполнению каталога и базы товаров с использованием новой рабочей модели детекции YOLO (`models/yolo_label.pt`), валидация процесса очистки базы/медиа и оптимизация мониторинга пакетного импорта.

### Выполненные задачи и изменения
1. **Верификация команд очистки и наполнения:**
   - Проверена и зафиксирована процедура полной очистки базы и медиа-файлов:
     `docker compose exec api python3 scripts/clear_database.py --yes` (TRUNCATE CASCADE таблиц `products`, `product_embeddings`, `import_jobs`, `import_items` и очистка `media/products`).
   - Проведён успешный пилотный импорт 5 товаров через `scripts/import_from_csv.py`.
2. **Доработка скрипта прямого импорта (`scripts/import_from_csv.py`):**
   - Добавлен замер времени выполнения (`time.perf_counter()`) для каждого обрабатываемого элемента.
   - Логирование дополнено временем обработки позиции:
     - `[{idx}/{total}] OK (0.xxs): id=... | title (manufacturer)`
     - `[{idx}/{total}] ERROR (0.xxs) title: error`
     - `[{idx}/{total}] SKIP/FAIL (0.00s) title: image not found`
   - Добавлен итоговый расчет суммарного времени импорта и средней скорости на позицию (`Total time: Xs, avg: Ys/item`).
   - Код верифицирован `ruff check`.
3. **Фиксация в Git:**
   - Коммит `4229585`: `Log item processing duration and total elapsed time in direct CSV import`.


### Точка продолжения

1. На сервере обновиться до `d845b08` или новее.
2. Выполнить `docker compose up -d --build`.
3. Проверить migration `20260916_0002`, `/api/ready`, `/api/imports` в OpenAPI.
4. Получить фактический dataset заказчика.
5. Проверить его поля и структуру изображений по `docs/BATCH_IMPORT.md`.
6. Подготовить архив через `scripts/prepare_dataset_import.sh`.
7. Запустить пробный batch на нескольких товарах и проверить products, embeddings, media, ошибки, качество поиска и latency.
8. После успешного пилота загрузить полный dataset и провести quality/load validation.

### Основные документы

- `md/START.md` — короткий первый запуск.
- `md/SERVER_COMMANDS.md` — расширенные серверные команды.
- `docs/BATCH_IMPORT.md` — формат и запуск пакетного импорта.
- `docs/PROD_BUILD_GUIDE.md` — production build.
- `docs/RUNBOOK.md` — эксплуатация.
- `AGENTS.md` — обязательные правила проекта и delivery workflow.

---

## Сессия 2026-09-21

### Цель
Создание поискового пайплайна v2 и системы выпрямления этикеток (нормализация геометрических и цилиндрических искажений бутылок в каноническую матрицу 256×256), исключение обрезки полезной площади контура сегментации и запуск визуальной диагностики.

### Реализовано и зафиксировано:

1. **Сервис сегментации и инференса:**
   - Сервис `SegmenterService` (`app/services/segmenter.py`) на базе модели `models/yolo_seg.pt`.
   - Загрузка модели через Git LFS, динамический выбор порога уверенности (`conf_seg`).

2. **Полноконтурная развёртка в каноническую матрицу 256×256 (Coons Patch):**
   - Отказ от `cv2.approxPolyDP` и `minAreaRect`, которые отрезали края этикеток и срезали надписи/узоры.
   - Определение 4 опорных точек границы по естественной ориентации кадра: $TL=\min(X+Y)$, $TR=\max(X-Y)$, $BR=\max(X+Y)$, $BL=\min(X-Y)$, что устранило поворот горизонтальных этикеток на 90°.
   - Разбиение непрерывного контура маски на 4 краевые кривые: верхняя дуга ($TL \to TR$), правый фланг ($TR \to BR$), нижняя дуга ($BL \to BR$), левый фланг ($TL \to BL$).
   - Применение трансфинитной интерполяции Coons Patch с 3D-тензорным бродкастингом (`U` (1, W, 1) и `V` (H, 1, 1)).
   - Цилиндрическая угловая коррекция сжатия текста по краям ($\sin\theta / \sin\theta_0$).
   - Развёртка через `cv2.remap` обеспечивает сохранение 100% площади обводки маски и точное совпадение краев матрицы с экстремумами контура.

3. **База данных и поиск v2:**
   - Таблица `product_embeddings_v2` (модель `ProductEmbeddingV2`, репозиторий `nearest_v2`, Alembic миграция `20260921_0005`).
   - Пайплайн `SearchPipelineV2` (`/api/v2/search`, `/api/v2/search-from-crop`, `/api/v2/eval/predict`).
   - Скрипт `scripts/build_embeddings_v2.py`: чистые эталоны каталога без сегментации размножаются через 116 аугментаций и векторизуются DINOv2.
   - Скрипт единого бенчмарка `scripts/run_benchmark.py` для замеров Hit@1 и latency v1 vs v2.

4. **Веб-интерфейс и инструмент диагностики:**
   - Страница `/rectify-test` — интерактивный визуализатор каскада с послойным включением BBox, сегментации, 4 углов и просмотром матрицы 256×256.
   - Страница `/search-v2` — визуальный поиск v2 с цепочкой карточек: «Оригинал ➔ BBox кроп ➔ Матрица v2 ➔ Top-1 победитель».
   - Кнопка «Сохранить диагностику в файл» (`/api/rectify/export_debug`):
     - Формирует и скачивает композитный коллаж высокого разрешения (`collage.png`).
     - Сохраняет на сервере в `media/debug_exports/<name>/` комплект: `01_overlay_full.png`, `02_bbox_crop.png`, `03_matrix_256.png`, подробный текстовый отчёт `diagnostic_report.txt` и `meta.json` со всеми BBox и масками.

