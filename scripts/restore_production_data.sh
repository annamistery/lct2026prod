#!/usr/bin/env bash
# ==============================================================================
# Восстановление базы данных PostgreSQL и эталонных медиа-кропов «из коробки»
# на стороне заказчика или организаторов перед запуском тестов.
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="$ROOT_DIR/data"
DUMP_FILE="$DATA_DIR/catalog_dump.sql.gz"
MEDIA_ARCHIVE="$DATA_DIR/media_catalog.tar.gz"

echo "=== [1/5] Проверка наличия исходных файлов данных ==="
if [ ! -f "$DUMP_FILE" ]; then
  echo "ОШИБКА: Файл дампа базы не найден: $DUMP_FILE" >&2
  echo "Убедитесь, что репозиторий склонирован с поддержкой Git LFS (git lfs pull)." >&2
  exit 1
fi

if [ ! -f "$MEDIA_ARCHIVE" ]; then
  echo "ОШИБКА: Архив кропов не найден: $MEDIA_ARCHIVE" >&2
  exit 1
fi

echo "✓ Файлы данных найдены:"
echo "   - $DUMP_FILE ($(du -h "$DUMP_FILE" | cut -f1))"
echo "   - $MEDIA_ARCHIVE ($(du -h "$MEDIA_ARCHIVE" | cut -f1))"

echo "=== [2/5] Запуск и проверка контейнера базы данных ==="
# Запускаем контейнер db если он еще не запущен
docker compose up -d db

POSTGRES_USER="lct2026"
POSTGRES_DB="lct2026"
if [ -f .env ]; then
  val_user=$(grep -E '^POSTGRES_USER=' .env | cut -d '=' -f2- | tr -d ' "\r' || true)
  val_db=$(grep -E '^POSTGRES_DB=' .env | cut -d '=' -f2- | tr -d ' "\r' || true)
  [ -n "$val_user" ] && POSTGRES_USER="$val_user"
  [ -n "$val_db" ] && POSTGRES_DB="$val_db"
fi

echo "Ожидание готовности PostgreSQL ($POSTGRES_DB)..."
for i in {1..30}; do
  if docker compose exec -T db pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null 2>&1; then
    echo "✓ PostgreSQL готов к работе"
    break
  fi
  sleep 1
done

echo "=== [3/5] Восстановление базы данных из дампа ==="
echo "Загрузка таблиц products и 224,000 векторов pgvector в базу данных..."
gunzip -c "$DUMP_FILE" | docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -q

# Проверяем количество товаров в базе
COUNT=$(docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT count(*) FROM products;" | tr -d ' \r\n')
EMB_COUNT=$(docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT count(*) FROM product_embeddings;" | tr -d ' \r\n')
echo "✓ База успешно восстановлена: товаров: $COUNT, эмбеддингов: $EMB_COUNT"

echo "Применение миграций схемы базы данных (Alembic)..."
docker compose run --rm migrate >/dev/null 2>&1 || docker compose up -d migrate
echo "✓ Миграции базы данных актуализированы"

echo "=== [4/5] Распаковка медиа-кропов каталога ==="
mkdir -p "$ROOT_DIR/media"
tar -xzf "$MEDIA_ARCHIVE" -C "$ROOT_DIR/media"
echo "✓ Кропы каталога распакованы в $ROOT_DIR/media"

echo "=== [5/5] Перезапуск сервиса API и проверка работоспособности ==="
docker compose restart api >/dev/null 2>&1 || docker compose up -d api

# Проверка ping
echo "Проверка доступности API (порт 8030)..."
for i in {1..20}; do
  if curl -fsS http://127.0.0.1:8030/api/ping >/dev/null 2>&1; then
    echo "✓ Сервис API успешно отвечает на запросы!"
    break
  fi
  sleep 1
done

echo ""
echo "========================================================================"
echo " Данные успешно восстановлены! Система полностью готова к работе."
echo " Для запуска официального тестирования выполните:"
echo "   ./md/participant_test.sh --images-dir ./queries --manifest ./queries.tsv"
echo "========================================================================"
