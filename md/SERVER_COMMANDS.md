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

## 3a. Собрать сведения о моделях старого проекта

Если модели ещё не перенесены из `$HOME/LCT2026`, обновить репозиторий и запустить единый безопасный скрипт инвентаризации:

```bash
git pull --ff-only
chmod +x scripts/collect_model_inventory.sh
./scripts/collect_model_inventory.sh "$HOME/LCT2026" model_inventory.txt
```

Проверяем:

```bash
test -s model_inventory.txt && echo 'INVENTORY OK'
wc -l model_inventory.txt
```

Получаем: файл `model_inventory.txt` со сведениями о DINO/LoRA, Hugging Face snapshot, старых индексах, каталогах и числе изображений. Скрипт не читает `.env`, process environment и секреты. Содержимое можно передать разработчику:

```bash
cat model_inventory.txt
```

Файл отчёта исключён из Git.

## 4. Разместить модели

Ожидаемая структура:

```text
models/
├── yolo_label.pt
├── dinov2-small/
└── dinov2_label_finetuned/
```

Проверяем:

```bash
test -f models/yolo_label.pt && echo 'YOLO OK'
test -d models/dinov2-small && echo 'DINO base OK'
test -d models/dinov2_label_finetuned && echo 'DINO adapter/model OK'
```

Получаем: три строки `OK`. Runtime монтирует `models/` только для чтения.

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

Проверяем: ответ HTTP 201 содержит UUID, `image_url`, `label_url` и имя embedding model.

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
