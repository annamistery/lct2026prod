# Runbook

## Диагностика

```bash
docker compose ps
docker compose logs -f api
docker compose logs -f db
curl http://localhost:8030/api/ping
curl http://localhost:8030/api/ready
```

`ping` проверяет процесс. `ready` требует PostgreSQL/pgvector и загруженные YOLO/DINO.

## Миграции

```bash
docker compose run --rm migrate alembic current
docker compose run --rm migrate alembic upgrade head
```

Откат сначала проверяется на копии данных. Изменять схему через `create_all` запрещено.

## Backup

PostgreSQL dump и каталог `media/` резервируются в одной точке согласованности. Восстановление только одного из них создаёт битые ссылки изображений.

## Модели

`models/` read-only. Для обновления разместить новую проверенную версию на хосте, проверить checksum и перезапустить API. При неготовых моделях `/api/ready` возвращает 503.

## Перегрузка

При GPU OOM уменьшить `GPU_CONCURRENCY`. При высокой SIFT latency уменьшить `CANDIDATE_POOL_SIZE` после проверки качества. Контролировать размер `media/` и rate limits Apache.
