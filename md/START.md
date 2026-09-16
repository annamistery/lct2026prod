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

Заменить `POSTGRES_PASSWORD` и тот же пароль внутри `DATABASE_URL`. Если Apache отдаёт web и `/api/` с одного адреса, оставить `CORS_ORIGINS=` пустым.

## 3. Проверить GPU и Compose

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
docker compose config --quiet
```

Результат: GPU виден внутри контейнера, Compose не выводит ошибок.

## 4. Собрать и запустить всё

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
{"ready":true,"database":true,"models":true}
{"products":[]}
20260916_0001 (head)
```

На этом чистый проект развёрнут. БД и media пока пустые; следующим этапом они наполняются из датасета заказчика.
