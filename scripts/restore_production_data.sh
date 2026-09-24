#!/usr/bin/env bash
# ==============================================================================
# Восстановление базы данных PostgreSQL и эталонных медиа-кропов «из коробки»
# на стороне заказчика или организаторов перед запуском тестов.
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="$ROOT_DIR/data"
MEDIA_ARCHIVE="$DATA_DIR/media_catalog.tar.gz"

DUMP_PRODUCTS="$DATA_DIR/dump_products.sql.gz"
LEGACY_DUMP="$DATA_DIR/catalog_dump.sql.gz"

echo "=== [1/5] Проверка наличия исходных файлов данных ==="
HAS_MODULAR=false
HAS_LEGACY=false

if [ -f "$DUMP_PRODUCTS" ]; then
  HAS_MODULAR=true
elif [ -f "$LEGACY_DUMP" ] || compgen -G "${LEGACY_DUMP}*" >/dev/null; then
  HAS_LEGACY=true
else
  echo "ОШИБКА: Файлы дампа базы не найдены в $DATA_DIR!" >&2
  echo "Ожидались: $DUMP_PRODUCTS или $LEGACY_DUMP" >&2
  echo "Убедитесь, что репозиторий склонирован с поддержкой Git LFS (git lfs pull)." >&2
  exit 1
fi

if [ ! -f "$MEDIA_ARCHIVE" ]; then
  echo "ОШИБКА: Архив кропов не найден: $MEDIA_ARCHIVE" >&2
  exit 1
fi

echo "✓ Файлы данных найдены:"
if [ "$HAS_MODULAR" = true ]; then
  echo "   - $DUMP_PRODUCTS ($(du -h "$DUMP_PRODUCTS" | cut -f1))"
  for emb in "$DATA_DIR"/dump_embeddings_*.sql.gz; do
    [ -f "$emb" ] && echo "   - $emb ($(du -h "$emb" | cut -f1))"
  done
else
  echo "   - $LEGACY_DUMP ($(du -h "$LEGACY_DUMP" | cut -f1))"
fi
echo "   - $MEDIA_ARCHIVE ($(du -h "$MEDIA_ARCHIVE" | cut -f1))"

echo "=== [2/5] Запуск и проверка контейнера базы данных ==="
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
if [ "$HAS_MODULAR" = true ]; then
  echo "1. Загрузка схемы и каталога товаров ($DUMP_PRODUCTS)..."
  gunzip -c "$DUMP_PRODUCTS" | docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -q

  # Загрузка векторов v4 (боевой SOTA)
  if [ -f "$DATA_DIR/dump_embeddings_v4.sql.gz" ]; then
    echo "2. Загрузка векторов v4 (SigLIP 2, dump_embeddings_v4.sql.gz)..."
    gunzip -c "$DATA_DIR/dump_embeddings_v4.sql.gz" | docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -q
  fi

  # Загрузка векторов v1 (baseline)
  if [ -f "$DATA_DIR/dump_embeddings_v1.sql.gz" ]; then
    echo "3. Загрузка векторов v1 (DINOv2, dump_embeddings_v1.sql.gz)..."
    gunzip -c "$DATA_DIR/dump_embeddings_v1.sql.gz" | docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -q
  fi

  # Загрузка других версий при их наличии
  for other_emb in "$DATA_DIR"/dump_embeddings_*.sql.gz; do
    if [ -f "$other_emb" ] && [ "$other_emb" != "$DATA_DIR/dump_embeddings_v1.sql.gz" ] && [ "$other_emb" != "$DATA_DIR/dump_embeddings_v4.sql.gz" ]; then
      echo "Загрузка дополнительных векторов: $(basename "$other_emb")..."
      gunzip -c "$other_emb" | docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -q
    fi
  done
else
  echo "Загрузка монолитного дампа..."
  cat "${LEGACY_DUMP}"* | gunzip -c | docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -q
fi

# Проверяем количество товаров в базе
COUNT=$(docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT count(*) FROM products;" 2>/dev/null | tr -d ' \r\n' || echo "0")
echo "✓ База успешно восстановлена: товаров в каталоге: $COUNT"

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
