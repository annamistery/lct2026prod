# Сканер Российских вин

Сервис распознаёт вино по фотографии этикетки и находит его в каталоге. К найденному вину AI-сомелье подбирает блюда, а к блюду — вина.

- **Backend:** FastAPI, один процесс на порту `8030`: API, веб-страницы и Swagger.
- **База данных:** PostgreSQL 16 + pgvector. В ней лежит каталог (2017 вин) и векторы изображений: v1 DINOv2 и v4 SigLIP 2.
- **Поиск:** каскад. DINOv2 отбирает 30 кандидатов, SigLIP 2 и DINOv2 вместе выбирают вино; ответ — «найдено», «похоже» или «Данного вина нет в каталоге». Точность top-1 на 128 размеченных фото — 95,3 %, ответ «нет в каталоге» ни разу не был ошибочным ([`docs/RECOGNITION_QUALITY.md`](docs/RECOGNITION_QUALITY.md)).
- **AI-сомелье:** к распознанному вину — справка «с чем подать» и похожие вина-альтернативы из каталога.
- **Поставка:** модели, дамп базы и картинки каталога лежат в репозитории через Git LFS. Приложение разворачивается в Docker без обучения и ручной загрузки данных.

Устройство проекта и описание модулей — в [`Architecture.md`](Architecture.md).

---

## 1. Ссылки приложения

После запуска все ссылки открываются по адресу `http://<адрес-сервера>:8030`. На том же компьютере это `http://localhost:8030`.

| Что | Ссылка |
|---|---|
| **Сканер этикеток (основной, каскад)** | http://localhost:8030/search-cascade |
| Мобильный сканер каскада (камера телефона) | http://localhost:8030/mobi-cascade |
| Автономный мобильный сканер | http://localhost:8030/export/mobitest_cascade.html |
| Мобильный сканер «Своё Вино» (каскад, сомелье, оценка) | http://localhost:8030/mobi/ |
| Каталог вин | http://localhost:8030/products |
| Добавление вина в каталог | http://localhost:8030/add |
| Пакетный импорт каталога | http://localhost:8030/import |
| Похожие этикетки («двойники») | http://localhost:8030/twins |
| Сканер v4 (SigLIP 2, диагностика) | http://localhost:8030/search-v4 |
| **Swagger — документация и проверка API** | http://localhost:8030/api/docs |
| Проверка «процесс жив» | http://localhost:8030/api/ping |
| **Проверка готовности** | http://localhost:8030/api/ready |

Основной эндпоинт для автоматической проверки: `POST http://localhost:8030/api/cascade/predict`. Изображение передаётся в multipart-поле `image`, ответ: `{"slug": "...", "status": "found" | "probable" | "not_in_catalog", "confidence": 0.93}`. Если вина нет в каталоге, `slug` = `null`. Подробнее — [`docs/SEARCH_CASCADE_API.md`](docs/SEARCH_CASCADE_API.md).

Камера в мобильных сканерах работает только по `https://` или на `localhost`. Чтобы открыть сканер с телефона по IP-адресу, опубликуйте сервис через HTTPS, например через Apache или nginx с сертификатом.

---

## 2. Логины и пароли

У приложения нет пользователей и входа по паролю: веб-страницы и API открыты. Пароль нужен только для базы данных PostgreSQL. Он задаётся в файле `.env`, который создаётся из `.env.example`.

| Параметр | Значение по умолчанию | Где задаётся |
|---|---|---|
| Пользователь PostgreSQL | `lct` | `.env` → `POSTGRES_USER` |
| **Пароль PostgreSQL** | `change-me` | `.env` → `POSTGRES_PASSWORD` **и** внутри `DATABASE_URL` |
| Имя базы | `lct2026` | `.env` → `POSTGRES_DB` |
| Строка подключения API | `postgresql+asyncpg://lct:change-me@db:5432/lct2026` | `.env` → `DATABASE_URL` |
| Хост / порт БД | `db:5432` (внутри Docker, наружу не публикуется) | `compose.yaml` |

Внешние ключи и токены для работы приложения не нужны: сомелье работает по шаблонам, без LLM.

**Правила для пароля:**

1. Пароль в `POSTGRES_PASSWORD` должен совпадать с паролем в `DATABASE_URL`. Если они разные, контейнер `migrate` упадёт с ошибкой `password authentication failed for user "lct"`.
2. PostgreSQL запоминает пароль **только при первом создании базы**. Поэтому пароль нужно задать в `.env` **до первого запуска**. Как сменить его потом — в разделе «Решение проблем».
3. На сервере, доступном из интернета, замените `change-me` на свой пароль. Файл `.env` в Git не коммитится.

---

## 3. Требования

| | Linux-сервер (рекомендуется) | Windows 10/11 |
|---|---|---|
| ОС | Ubuntu 22.04 / 24.04 | Windows 10/11 x64 |
| Docker | Docker Engine + Compose v2 | Docker Desktop (WSL 2) |
| GPU | NVIDIA + драйвер + NVIDIA Container Toolkit | NVIDIA + свежий драйвер (GPU доступен в Docker Desktop через WSL 2) |
| Прочее | `git`, `git-lfs`, `curl` | Git for Windows (Git Bash, включает Git LFS) |
| Диск | ≈ 15 ГБ: образ ≈ 10 ГБ + данные ≈ 3 ГБ | то же |

Порт `8030` должен быть свободен.

---

## 4. Установка «из коробки»

Все команды выполняются в `bash`. На Linux это обычный терминал, на Windows — **Git Bash**.

### 4.1 Скачать проект, модели и данные

```bash
git clone https://github.com/vadfe/lct2026prod.git
cd lct2026prod
git lfs install
git lfs pull
sha256sum --check models/MODEL_MANIFEST.sha256
```

Проверка контрольных сумм должна вывести `OK` по каждой модели. В папке `data/` должны появиться `dump_products.sql.gz`, `dump_embeddings_v1.sql.gz`, `dump_embeddings_v4.sql.gz` и `media_catalog.tar.gz` — всего около 1,4 ГБ.

### 4.2 Создать конфигурацию

```bash
cp .env.example .env
```

Этого достаточно для запуска с паролем по умолчанию `change-me`. Чтобы задать свой пароль, откройте `.env` (`nano .env` или любым редактором) и замените `change-me` **в двух местах**:

```
POSTGRES_PASSWORD=МойПароль
DATABASE_URL=postgresql+asyncpg://lct:МойПароль@db:5432/lct2026
```

Если веб-страницы и API открываются с одного адреса (обычный случай), оставьте `CORS_ORIGINS=` пустым.

**Только на Linux** — выдайте API права на запись в `media/`:

```bash
sed -i "s/^APP_UID=.*/APP_UID=$(id -u)/; s/^APP_GID=.*/APP_GID=$(id -g)/" .env
```

На Windows эту команду **не выполняйте**, оставьте `APP_UID=1000` и `APP_GID=1000`.

### 4.3 Проверить GPU (Linux)

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
docker compose config --quiet
```

### 4.4 Восстановить базу и картинки каталога

Выполняется **один раз** на чистой установке:

```bash
chmod +x scripts/*.sh
./scripts/restore_production_data.sh
```

Скрипт выполняет пять шагов:

1. Запускает PostgreSQL.
2. Загружает каталог и векторы v1/v4 — около 234 тысяч строк. Это занимает несколько минут.
3. Применяет миграции.
4. Распаковывает картинки каталога в `media/`.
5. Запускает API. При первом запуске собирается Docker-образ, это 10–20 минут.

### 4.5 Запустить приложение

```bash
docker compose up -d --build
docker compose ps
```

В выводе `docker compose ps` у `db` и `api` должно быть состояние `running`, а `migrate` должен завершиться с кодом `0`. Первая загрузка моделей занимает около минуты.

### 4.6 Проверить

```bash
curl http://127.0.0.1:8030/api/ping
curl http://127.0.0.1:8030/api/ready
```

Ожидаемый ответ:

```json
{"ok":true}
{"ready":true,"database":true,"models":true,"sommelier":true}
```

Проверка распознавания на одном фото:

```bash
curl -X POST http://127.0.0.1:8030/api/cascade/predict -F "image=@путь/к/фото.jpg"
```

Затем откройте в браузере **http://localhost:8030/search-cascade** и загрузите фото этикетки.

---

## 5. Управление

```bash
docker compose ps                     # состояние сервисов
docker compose logs -f api            # логи API
docker compose stop                   # остановить (данные сохраняются)
docker compose start                  # запустить снова
docker compose up -d --build          # пересобрать после изменения кода
docker compose restart api            # перезапуск API (например, после правки .env)
```

База хранится в Docker volume `lct2026prod_postgres_data`. Картинки лежат в папке `media/`. Команда `docker compose down -v` **удалит базу**, после неё нужно повторить шаг 4.4.

---

## 6. Тестирование распознавания

Контрольный набор состоит из папки `queries/` с фотографиями и файла `queries.tsv` с заголовком `query_id<TAB>image_path`.

```bash
./md/participant_test.sh --images-dir ./queries --manifest ./queries.tsv \
    --endpoint http://127.0.0.1:8030/api/cascade/predict --output predictions.jsonl
```

Для скрипта нужны `curl` и `jq`. На Windows `jq` устанавливается командой `winget install jqlang.jq`. Подробнее — в [`md/ИНСТРУКЦИЯ_ПРОВЕРКИ.md`](md/ИНСТРУКЦИЯ_ПРОВЕРКИ.md).

Если у набора есть правильные ответы, точность и зоны ответа (найдено / похоже / нет в каталоге) считает скрипт:

```bash
python scripts/eval_recognition.py --api http://127.0.0.1:8030 --sets tmp1=tmp/1 tmp2=tmp/2 "мои=путь/к/папке"
```

Разметка — `mapping.json` (формат тестовых пакетов) или `labels.tsv` с колонками `image_path`, `expected_slug`, `status` (`in_catalog` / `not_in_catalog` / `unsure`), `alt_slugs`. То же можно сделать в браузере: http://localhost:8030/search-cascade → «Мой набор» → папка с фото и `labels.tsv` → «Проверить все по списку».

Проверки кода для разработчика:

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
```

---

## 7. Обновление каталога вин

Каталог можно дополнить своими винами:

1. Положите CSV в `./imports/wines_integrated_cleared.csv`. Колонки: `Название вина`, `Винодельня`, `Описание`, `Slug`, `Название фото`.
2. Положите фото бутылок в `./media/catalog_sources/`.
3. Запустите загрузку:

```bash
# проверка без записи
docker compose exec api python scripts/ingest_cascade_catalog.py \
    --csv /imports/wines_integrated_cleared.csv --images-dir /media/catalog_sources --dry-run

# загрузка (повторный запуск пропускает уже загруженные вина)
docker compose exec api python scripts/ingest_cascade_catalog.py \
    --csv /imports/wines_integrated_cleared.csv --images-dir /media/catalog_sources
```

Полезные флаги: `--limit 2` — пробный прогон на двух винах, `--force-rebuild` — полная перезапись. Для каждого вина скрипт находит этикетку (YOLO), делает 116 аугментаций и записывает векторы v1 и v4 в базу.

Если запускаете команды из **Git Bash**, добавляйте перед ними `MSYS_NO_PATHCONV=1`. Иначе Git Bash превратит пути `/imports/...` в пути Windows:

```bash
MSYS_NO_PATHCONV=1 docker compose exec api python scripts/ingest_cascade_catalog.py --csv /imports/wines_integrated_cleared.csv --images-dir /media/catalog_sources --dry-run
```

Пакетный импорт архивов через API описан в [`docs/BATCH_IMPORT.md`](docs/BATCH_IMPORT.md).

---

## 8. Решение проблем

**`service "migrate" didn't complete successfully: exit 1`, в логах `password authentication failed for user "lct"`**

База создана с другим паролем, чем указан сейчас в `.env`. Установите в базе пароль из `.env` — данные не пострадают:

```bash
PW=$(grep '^POSTGRES_PASSWORD=' .env | cut -d= -f2-)
docker compose exec db psql -U lct -d lct2026 -c "ALTER USER lct PASSWORD '$PW';"
docker compose up -d
```

Логи миграции можно посмотреть командой `docker compose logs migrate`.

**`/api/ready` возвращает 503 или `"models":false`**

Модели не скачаны из Git LFS или ещё загружаются. Выполните `git lfs pull`, проверьте `sha256sum --check models/MODEL_MANIFEST.sha256`, затем посмотрите `docker compose logs api`.

**`Bind for 0.0.0.0:8030 failed: port is already allocated`**

Порт занят другим приложением или другой копией проекта. Найдите, кто его занимает (`docker ps`), и остановите: `docker compose stop api` в папке той копии.

**`"database":false`**

База не запущена или пустая. Проверьте `docker compose ps db` и повторите шаг 4.4.

**В каталоге нет картинок**

Картинки не распакованы. Выполните `tar -xzf data/media_catalog.tar.gz -C media`.

**Не хватает видеопамяти (CUDA out of memory)**

Закройте другие программы, которые используют GPU, в том числе вторую копию проекта. Проверьте, что в `.env` стоит `GPU_CONCURRENCY=1`, и выполните `docker compose restart api`.

---

## 9. Доступ по ссылке для коллег

Компьютер с видеокартой работает как сервер API: распознаёт фото по запросам. Коллеги открывают сканер по HTTPS-ссылке, роутер настраивать не нужно.

```bash
docker compose --profile public up -d   # публичный вход (порт 8031) + туннель localhost.run
bash scripts/public_url.sh              # текущие ссылки: сканер, каскадный поиск, API, Swagger
```

- Публичный вход (`deploy/public/nginx.conf`) пропускает только сканер, распознавание, сомелье и просмотр каталога. Добавление и импорт вин (`/add`, `/import`, `POST /api/products`, `/api/imports`) по ссылке закрыты и доступны только на `http://localhost:8030`.
- Адрес туннеля меняется после переподключения — запустите `scripts/public_url.sh` ещё раз и отправьте новую ссылку.
- Альтернатива — Cloudflare Tunnel: `docker compose --profile cloudflare up -d` (в некоторых сетях заблокирован).
- Остановить: `docker compose --profile public stop public tunnel`.

**Сканер на GitHub Pages.** Workflow `.github/workflows/pages.yml` публикует `web/` статикой. Адрес API передаётся в ссылке: `https://<владелец>.github.io/lct2026prod/?api=https://<адрес туннеля>`. Сайт Pages должен быть в `CORS_ORIGINS` (`.env`). Один раз владелец репозитория включает Pages: Settings → Pages → Source: GitHub Actions.

## Документация

- [`Architecture.md`](Architecture.md) — устройство проекта, модули, запуск, тестирование.
- [`docs/SEARCH_CASCADE_API.md`](docs/SEARCH_CASCADE_API.md) — API каскадного поиска и зоны ответа.
- [`docs/RECOGNITION_QUALITY.md`](docs/RECOGNITION_QUALITY.md) — точность, калибровка «нет в каталоге», проблемы данных каталога.
- [`docs/SEARCH_V4_API.md`](docs/SEARCH_V4_API.md) — API поиска v4.
- [`docs/SOMMELIER.md`](docs/SOMMELIER.md) — AI-сомелье.
- [`docs/BATCH_IMPORT.md`](docs/BATCH_IMPORT.md) — пакетный импорт.
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — эксплуатация, бэкапы, диагностика.
- [`md/SERVER_COMMANDS.md`](md/SERVER_COMMANDS.md) — команды для сервера.
