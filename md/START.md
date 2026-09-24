# Первый запуск на сервере

Короткая инструкция для чистого Ubuntu-сервера с Docker, NVIDIA Container Toolkit и Git LFS.

## 1. Скачать проект и модели

```bash
cd ~
git clone https://github.com/vadfe/lct2026prod.git
cd lct2026prod
git lfs pull
sha256sum --check models/MODEL_MANIFEST.sha256
```

Результат: все строки проверки моделей имеют статус `ЦЕЛ`/`OK`.

## 2. Создать конфигурацию

```bash
cp .env.example .env
nano .env
```

Заменить `POSTGRES_PASSWORD` и тот же пароль внутри `DATABASE_URL`. Установить `APP_UID` и `APP_GID` по выводу `id -u` и `id -g`, чтобы API мог писать в host-каталог `media/`. Если Apache отдаёт web и `/api/` с одного адреса, оставить `CORS_ORIGINS=` пустым.

```bash
sed -i "s/^APP_UID=.*/APP_UID=$(id -u)/; s/^APP_GID=.*/APP_GID=$(id -g)/" .env
```

## 3. Проверить GPU и Compose

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
docker compose config --quiet
```

Результат: GPU виден внутри контейнера, Compose не выводит ошибок.

## 4. Восстановить боевые данные (каталог и векторы) и запустить сервисы

Для запуска системы «из коробки» с уже готовой полной базой товаров и предрассчитанными векторами:

```bash
chmod +x scripts/*.sh
./scripts/restore_production_data.sh
```

Скрипт автоматически поднимет `db`, восстановит таблицы и pgvector индексы, применит миграции Alembic и распакует медиа-кропы в `media/`.

После этого соберите и запустите всё окружение:
```bash
docker compose up -d --build
docker compose ps
docker compose logs --tail=100 migrate
docker compose logs --tail=200 api
```

Результат: `db` и `api` запущены, `migrate` завершён с кодом 0. При первом запуске загрузка моделей может занять время.


## 5. Проверить приложение

```bash
curl -fsS http://127.0.0.1:8030/api/ping; echo
curl -fsS http://127.0.0.1:8030/api/ready; echo
curl -fsS http://127.0.0.1:8030/api/products; echo
docker compose run --rm migrate alembic current
```

Ожидается:

```text
{"ok":true}
{"ready":true,"database":true,"models":true,"sommelier":true}
{"products":[]}
20260916_0001 (head)
```

На этом чистый проект развёрнут. БД и media пока пустые. Сомелье (`sommelier: true` выше) от БД не зависит,
собирается из своего CSV при старте API и уже готов:

```bash
curl -fsS -X POST http://127.0.0.1:8030/api/sommelier/ask -H 'Content-Type: application/json' -d '{"message": "стейк рибай"}'
```

Подробности API и точка интеграции с распознаванием этикеток: `docs/SOMMELIER.md`.

## 6. Загрузить датасет заказчика

```bash
cp /путь/customer.zip imports/inbox/
./scripts/prepare_dataset_import.sh imports/inbox/customer.zip customer-001
curl -fsS -X POST http://127.0.0.1:8030/api/imports -H 'Content-Type: application/json' -d '{"batch_id":"customer-001","manifest_name":"manifest.json"}'
```

API вернёт `job_id`. Проверка прогресса:

```bash
curl -fsS http://127.0.0.1:8030/api/imports/<job_id>
```

Форматы JSON/CSV и структура архива: `docs/BATCH_IMPORT.md`.
