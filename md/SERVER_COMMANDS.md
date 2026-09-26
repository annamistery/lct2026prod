# Команды для production-сервера

Краткий чек-лист для Ubuntu-сервера с NVIDIA GPU. Все команды выполняются из каталога проекта, если не указано иное.

## 1. Проверить сервер

```bash
nvidia-smi
docker --version
docker compose version
git lfs version
```

Проверяем: GPU и NVIDIA driver видны, установлены Docker Compose и Git LFS.

Получаем: четыре команды завершаются без ошибок; `nvidia-smi` показывает GPU.

Если `git lfs version` отвечает, что `lfs` не является командой Git, установить и активировать Git LFS:

```bash
sudo apt update
sudo apt install -y git-lfs
git lfs install
git lfs version
```

Получаем: строку `git-lfs/3.x.x`. После этого продолжить со шага 2.

Для RTX 5070 Ti драйвер с CUDA 13.0 подходит для запуска контейнера с CUDA 12.4: новый NVIDIA driver обратно совместим с более старым CUDA runtime внутри контейнера.

## 2. Скачать проект

```bash
git clone https://github.com/vadfe/lct2026prod.git
cd lct2026prod
git lfs pull
git status
```

Проверяем: репозиторий скачан, LFS-файлы получены, рабочее дерево чистое.

Получаем: `working tree clean`.

## 3. Подготовить конфигурацию

```bash
cp .env.example .env
nano .env
```

Обязательно изменить:

```dotenv
POSTGRES_PASSWORD=<сложный-пароль>
DATABASE_URL=postgresql+asyncpg://lct:<тот-же-пароль>@db:5432/lct2026
CORS_ORIGINS=https://<домен-заказчика>
```

Проверяем:

```bash
grep -E '^(ENVIRONMENT|DATABASE_URL|CORS_ORIGINS|DINO_MODEL_PATH|DINO_BASE_MODEL_PATH|YOLO_MODEL_PATH)=' .env
```

Получаем: production-значения и пути внутри контейнера. Не публикуйте вывод с паролем.

## 3a. Один раз перенести полный комплект production-моделей в Git

Этот шаг выполняется на сервере, где существуют старый проект `$HOME/LCT2026` и Hugging Face cache. Скрипт переносит только модели; изображения, кропы, датасеты и старые индексы не копируются.

```bash
git pull --ff-only
./scripts/stage_production_models.sh "$HOME/LCT2026"
```

Скрипт проверяет и копирует:

```text
models/yolo_label.pt
web/models/yolov8_label.onnx
models/dinov2-small/
models/dinov2_label_finetuned/    # финальный dino_aug_116
models/MODEL_MANIFEST.sha256
```

После проверки checksum скрипт спрашивает подтверждение commit/push. Ответить `y`. Получаем отдельный Git commit со всеми production-моделями в Git LFS.

Не добавлять в Git каталожные изображения и кропы: на следующем этапе PostgreSQL и файловое хранилище наполняются из датасета заказчика.

## 4. Проверить модели после чистого clone

```bash
git lfs pull
sha256sum --check models/MODEL_MANIFEST.sha256
test -f models/yolo_label.pt && echo 'YOLO OK'
test -f web/models/yolov8_label.onnx && echo 'ONNX OK'
test -f models/dinov2-small/model.safetensors && echo 'DINO base OK'
test -f models/dinov2_label_finetuned/adapter_model.bin && echo 'DINO adapter OK'
```

Получаем: все checksums имеют статус `OK` и четыре строки проверки моделей. Runtime монтирует `models/` только для чтения.

## 5. Проверить доступ Docker к GPU

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Проверяем: GPU виден из контейнера.

Получаем: таблицу `nvidia-smi` без ошибки `could not select device driver`.

## 6. Проверить Compose-конфигурацию

```bash
docker compose config --quiet
```

Проверяем: `.env`, volumes, зависимости сервисов и YAML валидны.

Получаем: команда завершается с кодом 0 без вывода.

## 7. Собрать образы

```bash
docker compose build --pull
```

Проверяем: Python-зависимости и приложение собираются.

Получаем: успешно собранный image API.

## 8. Запустить PostgreSQL и миграции

```bash
docker compose up -d db
docker compose run --rm migrate
```

Проверяем:

```bash
docker compose exec db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
docker compose run --rm migrate alembic current
```

Если переменные shell не экспортированы, используйте значения `POSTGRES_USER` и `POSTGRES_DB` из `.env` вручную.

Получаем: установленную версию pgvector и revision `20260916_0001 (head)`.

## 9. Запустить API

```bash
docker compose up -d api
docker compose ps
docker compose logs --tail=100 api
```

Проверяем: `db` и `api` имеют статус running/healthy, миграция завершилась успешно, модели загрузились без traceback.

## 10. Проверить health endpoints

```bash
curl -fsS http://127.0.0.1:8030/api/ping
curl -fsS http://127.0.0.1:8030/api/ready
```

Получаем:

```json
{"ok":true}
```

и readiness с `"ready": true`, `"database": true`, `"models": true`.

## 11. Добавить тестовый товар

```bash
curl -fsS -X POST http://127.0.0.1:8030/api/products \
  -F 'title=Тестовый товар' \
  -F 'manufacturer=Тестовый производитель' \
  -F 'description=Проверка production API' \
  -F 'image=@/полный/путь/к/фото-бутылки.jpg'
```

Проверяем: ответ HTTP 201 содержит UUID, текстовые поля товара, `image_url` и `label_url`. В БД создаются одна запись `products` и первый эталон в `product_embeddings` с типом `catalog`.

Затем:

```bash
curl -fsS http://127.0.0.1:8030/api/products
```

Получаем: созданный товар в каталоге.

## 12. Проверить поиск полного кадра

```bash
curl -fsS -X POST http://127.0.0.1:8030/api/v1/search \
  -F 'image=@/полный/путь/к/фото-бутылки.jpg' \
  -F 'k=5'
```

Проверяем: только versioned URL `/api/v1/search`; ответ содержит `winner`, `results`, DINO similarity, SIFT score/inliers и timings.

## 13. Проверить поиск готового кропа

```bash
curl -fsS -X POST http://127.0.0.1:8030/api/v1/search-from-crop \
  -F 'image=@/полный/путь/к/кропу-этикетки.jpg' \
  -F 'k=5'
```

Получаем: Top-K после pgvector candidate search и финального SIFT reranking.

## 14. Проверить Apache proxy

Через внешний HTTPS-домен:

```bash
curl -fsS https://<домен-заказчика>/api/ping
curl -fsS https://<домен-заказчика>/api/ready
```

Проверяем: Apache проксирует `/api/*`, клиент не обращается к внутреннему порту напрямую.

Затем открыть:

```text
https://<домен-заказчика>/<путь-к-web>/index.html
https://<домен-заказчика>/<путь-к-web>/add.html
https://<домен-заказчика>/api/docs
```

Получаем: камера работает в secure context, товар добавляется, поиск возвращает результат.

## 15. Посмотреть состояние и логи

```bash
docker compose ps
docker compose logs -f api
docker compose logs -f db
```

Остановить просмотр логов: `Ctrl+C`.

## 16. Обновить проект после нового push

```bash
git status
git pull --ff-only
git lfs pull
docker compose build
docker compose run --rm migrate
docker compose up -d api
docker compose ps
curl -fsS http://127.0.0.1:8030/api/ready
```

Проверяем: до `git pull` нет серверных изменений; миграции применены; API снова ready.

## 17. Backup PostgreSQL и media

```bash
mkdir -p backups
docker compose exec -T db pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc > "backups/lct2026_$(date +%F_%H%M).dump"
tar -czf "backups/media_$(date +%F_%H%M).tar.gz" media/
ls -lh backups/
```

Получаем: согласованную пару backup-файлов БД и media. Хранить и восстанавливать их вместе.

## 18. Остановить сервис

```bash
docker compose down
```

PostgreSQL volume сохраняется. Команду `docker compose down -v` не выполнять: она удаляет данные БД.

## 19. Раздельный экспорт production-данных и отправка в Git

Для модульного экспорта каталога товаров и версий эмбеддингов в `data/` выполните на сервере:

```bash
chmod +x scripts/*.sh

# Вариант А: Автоматический экспорт и пошаговый push в Git
./scripts/export_production_data.sh --push

# Вариант Б: Экспорт без авто-пуша (команды для отправки будут выведены на экран)
./scripts/export_production_data.sh
```

Скрипт формирует независимые компактные файлы под лимиты Git LFS:
- `data/dump_products.sql.gz` — схема и товары (~300 КБ)
- `data/dump_embeddings_v1.sql.gz` — DINOv2 baseline (~300 МБ)
- `data/dump_embeddings_v4.sql.gz` — SigLIP 2 SOTA (~600 МБ)
- `data/media_catalog.tar.gz` — кропы эталонов (~102 МБ)

При ручной отправке с сервера по частям:
```bash
# 1. Товары каталога
git add data/dump_products.sql.gz
git commit -m "data: update products catalog dump"
git push origin main

# 2. Векторы v1
git add data/dump_embeddings_v1.sql.gz
git commit -m "data: update v1 DINOv2 product embeddings dump"
git push origin main

# 3. Векторы v4
git add data/dump_embeddings_v4.sql.gz
git commit -m "data: update v4 SigLIP 2 product embeddings dump"
git push origin main
```



## 20. Управление каталогом и векторами каскада (Cascade v1 + v4)

Единый оптимизированный скрипт `scripts/ingest_cascade_catalog.py` выполняет один проход на товар:
YOLO-детекция → Letterbox 518×518 → 116 аугментаций в RAM → DINOv2 v1 (384d) + SigLIP 2 v4 (768d) → атомарная запись в БД.

Перед запуском CSV и исходные изображения должны быть доступны внутри контейнера (например, через `/imports` и `/media/catalog_sources`).

```bash
# 1. Проверить CSV и сопоставление изображений без записи в БД
docker compose exec api python scripts/ingest_cascade_catalog.py \
    --csv /imports/wines_integrated_cleared.csv \
    --images-dir /media/catalog_sources \
    --dry-run

# Список отсутствующих изображений с названием, slug, ссылкой и ожидаемым именем файла
# Результат сохранится в imports/missing_images.csv (доступен на хосте)
docker compose exec api python scripts/list_missing_catalog_images.py \
    --csv /imports/wines_integrated_cleared.csv \
    --images-dir /media/catalog_sources \
    --output /media/missing_images.csv

# 2. Тестовый прогон на 2 товарах
docker compose exec api python scripts/ingest_cascade_catalog.py \
    --csv /imports/wines_integrated_cleared.csv \
    --images-dir /media/catalog_sources \
    --limit 2

# 3. Полная заливка каталога
docker compose exec api python scripts/ingest_cascade_catalog.py \
    --csv /imports/wines_integrated_cleared.csv \
    --images-dir /media/catalog_sources

# 4. Принудительная перезапись всех векторов и медиа
docker compose exec api python scripts/ingest_cascade_catalog.py \
    --csv /imports/wines_integrated_cleared.csv \
    --images-dir /media/catalog_sources \
    --force-rebuild
```

Скрипт идемпотентен: повторный запуск пропускает товары, у которых уже есть ровно 116 векторов в обеих таблицах (`product_embeddings` и `product_embeddings_v4`).
При обрыве предыдущего запуска или частичных данных автоматически очищает «битые» векторы и перезаписывает их.

## 21. Запуск финального бенчмарка каскада (для отчёта заказчику)

Для комплексной проверки точности и скорости финального каскада по всем тестовым наборам (`tmp1`, `tmp2`, `imports`) и генерации итогового отчета в Markdown:

```bash
# Результаты сохраняются в /media/artifacts (видна на хосте)
docker compose exec api python scripts/eval_cascade_final.py \
    --packs 1 2
```

Для подбора порога уверенности, ниже которого `/api/cascade/predict` возвращает `slug: null`:

```bash
docker compose exec api python scripts/eval_cascade_final.py \
    --packs 1 2 \
    --threshold-sweep 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9
```

Результаты тестирования сохраняются в:
- `artifacts/cascade_final_report.md` — итоговая сводная таблица с разбивкой по точности, раннему выходу v1, арбитражу v4 и перцентилям задержек (P50, P90, P95).
- `artifacts/cascade_final_results.json` — детальные метрики по каждому проверенному изображению.

## 22. Мобильный сканер каскада

- **Внутренний интерфейс (через FastAPI):** `http://<server-ip>:8030/mobi-cascade` (или `http://<server-ip>:8030/export/mobitest_cascade.html`).
- **Автономный клиент (для внешнего веб-сервера / папки `export`):** скопируйте каталог `export/` на внешний сервер Nginx/Apache и укажите адрес бэкенда в `export/config.js` (`window.MOBI_API_BASE = "http://<server-ip>:8030"`).
