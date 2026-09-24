#!/usr/bin/env bash
# ==============================================================================
# Раздельный экспорт базы данных PostgreSQL по версиям и эталонных кропов
# для компактного хранения в Git LFS и модульного восстановления.
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="$ROOT_DIR/data"
mkdir -p "$DATA_DIR"

PUSH_TO_GIT=false
EXPORT_ALL_VERSIONS=false

for arg in "$@"; do
  case "$arg" in
    --push)
      PUSH_TO_GIT=true
      shift
      ;;
    --all-versions)
      EXPORT_ALL_VERSIONS=true
      shift
      ;;
    --help|-h)
      echo "Использование: $0 [--push] [--all-versions]"
      echo "  --push          Автоматически пошагово отправлять каждую партию в Git после экспорта"
      echo "  --all-versions  Экспортировать также архивные таблицы эмбеддингов v2 и v3"
      exit 0
      ;;
  esac
done

echo "=== [1/5] Проверка окружения и сервиса PostgreSQL ==="
if ! command -v docker >/dev/null 2>&1; then
  echo "ОШИБКА: docker не найден" >&2
  exit 1
fi

if ! docker compose ps --services --filter "status=running" | grep -q "^db$"; then
  echo "ОШИБКА: Сервис базы данных 'db' не запущен. Запустите: docker compose up -d db" >&2
  exit 1
fi

POSTGRES_USER="lct2026"
POSTGRES_DB="lct2026"
if [ -f .env ]; then
  val_user=$(grep -E '^POSTGRES_USER=' .env | cut -d '=' -f2- | tr -d ' "\r' || true)
  val_db=$(grep -E '^POSTGRES_DB=' .env | cut -d '=' -f2- | tr -d ' "\r' || true)
  [ -n "$val_user" ] && POSTGRES_USER="$val_user"
  [ -n "$val_db" ] && POSTGRES_DB="$val_db"
fi

echo "Используется база: '$POSTGRES_DB' (пользователь: '$POSTGRES_USER')"

echo "=== [2/5] Экспорт каталога товаров и схемы БД (dump_products.sql.gz) ==="
DUMP_PRODUCTS="$DATA_DIR/dump_products.sql.gz"
# Экспортируем схему и базовые таблицы: товары, миграции, задания импорта
docker compose exec -T db pg_dump \
  -U "$POSTGRES_USER" \
  --clean \
  --if-exists \
  --no-owner \
  --no-privileges \
  -t products \
  -t import_jobs \
  -t import_items \
  -t alembic_version \
  "$POSTGRES_DB" | gzip -9 > "$DUMP_PRODUCTS"

PROD_SIZE=$(du -h "$DUMP_PRODUCTS" | cut -f1)
echo "✓ Дамп товаров успешно создан: $DUMP_PRODUCTS ($PROD_SIZE)"

echo "=== [3/5] Экспорт векторов эмбеддингов по версиям ==="

# v1: DINOv2 baseline (vector 384)
DUMP_EMB_V1="$DATA_DIR/dump_embeddings_v1.sql.gz"
if docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1 FROM information_schema.tables WHERE table_name = 'product_embeddings';" | grep -q 1; then
  echo "Экспорт эмбеддингов v1 (product_embeddings, DINOv2)..."
  docker compose exec -T db pg_dump \
    -U "$POSTGRES_USER" \
    --data-only \
    --no-owner \
    --no-privileges \
    -t product_embeddings \
    "$POSTGRES_DB" | gzip -9 > "$DUMP_EMB_V1"
  V1_SIZE=$(du -h "$DUMP_EMB_V1" | cut -f1)
  echo "✓ Дамп эмбеддингов v1 создан: $DUMP_EMB_V1 ($V1_SIZE)"
else
  echo "Таблица product_embeddings не найдена, пропуск v1."
fi

# v4: SigLIP 2 SOTA (vector 768)
DUMP_EMB_V4="$DATA_DIR/dump_embeddings_v4.sql.gz"
if docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1 FROM information_schema.tables WHERE table_name = 'product_embeddings_v4';" | grep -q 1; then
  echo "Экспорт эмбеддингов v4 (product_embeddings_v4, SigLIP 2)..."
  docker compose exec -T db pg_dump \
    -U "$POSTGRES_USER" \
    --data-only \
    --no-owner \
    --no-privileges \
    -t product_embeddings_v4 \
    "$POSTGRES_DB" | gzip -9 > "$DUMP_EMB_V4"
  V4_SIZE=$(du -h "$DUMP_EMB_V4" | cut -f1)
  echo "✓ Дамп эмбеддингов v4 создан: $DUMP_EMB_V4 ($V4_SIZE)"
else
  echo "Таблица product_embeddings_v4 не найдена, пропуск v4."
fi

# Дополнительные архивные версии при запросе
if [ "$EXPORT_ALL_VERSIONS" = true ]; then
  for ver in v2 v3; do
    tbl="product_embeddings_${ver}"
    dump_file="$DATA_DIR/dump_embeddings_${ver}.sql.gz"
    if docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1 FROM information_schema.tables WHERE table_name = '$tbl';" | grep -q 1; then
      echo "Экспорт архивных эмбеддингов $ver ($tbl)..."
      docker compose exec -T db pg_dump \
        -U "$POSTGRES_USER" \
        --data-only \
        --no-owner \
        --no-privileges \
        -t "$tbl" \
        "$POSTGRES_DB" | gzip -9 > "$dump_file"
      echo "✓ Дамп $ver создан: $dump_file ($(du -h "$dump_file" | cut -f1))"
    fi
  done
fi

echo "=== [4/5] Упаковка эталонных кропов медиа-хранилища ==="
MEDIA_ARCHIVE="$DATA_DIR/media_catalog.tar.gz"
if [ -d "media/products" ]; then
  tar -czf "$MEDIA_ARCHIVE" -C media products
elif [ -d "media" ] && [ "$(ls -A media)" ]; then
  tar -czf "$MEDIA_ARCHIVE" -C media .
else
  echo "ВНИМАНИЕ: Папка media пуста или не найдена. Создан пустой архив."
  tar -czf "$MEDIA_ARCHIVE" -T /dev/null
fi
MEDIA_SIZE=$(du -h "$MEDIA_ARCHIVE" | cut -f1)
echo "✓ Архив кропов создан: $MEDIA_ARCHIVE ($MEDIA_SIZE)"

echo "=== [5/5] Проверка Git LFS ==="
if command -v git-lfs >/dev/null 2>&1; then
  git lfs install >/dev/null 2>&1 || true
  git lfs track "data/*.sql.gz" >/dev/null 2>&1 || true
  git lfs track "data/*.tar.gz" >/dev/null 2>&1 || true
fi

echo ""
echo "========================================================================"
echo " Экспорт успешно завершен!"
echo " Созданы файлы:"
echo "   - $DUMP_PRODUCTS ($PROD_SIZE)"
[ -f "$DUMP_EMB_V1" ] && echo "   - $DUMP_EMB_V1 ($V1_SIZE)"
[ -f "$DUMP_EMB_V4" ] && echo "   - $DUMP_EMB_V4 ($V4_SIZE)"
echo "   - $MEDIA_ARCHIVE ($MEDIA_SIZE)"
echo "========================================================================"

if [ "$PUSH_TO_GIT" = true ]; then
  echo ""
  echo "--- Отправка партий в Git (флаг --push активен) ---"
  
  echo "1. Отправка каталога товаров (dump_products.sql.gz)..."
  git add "$DUMP_PRODUCTS"
  git commit -m "data: update products catalog dump" || true
  git push origin main

  if [ -f "$DUMP_EMB_V1" ]; then
    echo "2. Отправка эмбеддингов v1 DINOv2 (dump_embeddings_v1.sql.gz)..."
    git add "$DUMP_EMB_V1"
    git commit -m "data: update v1 DINOv2 product embeddings dump" || true
    git push origin main
  fi

  if [ -f "$DUMP_EMB_V4" ]; then
    echo "3. Отправка эмбеддингов v4 SigLIP 2 (dump_embeddings_v4.sql.gz)..."
    git add "$DUMP_EMB_V4"
    git commit -m "data: update v4 SigLIP 2 product embeddings dump" || true
    git push origin main
  fi

  echo "4. Отправка медиа-архива (media_catalog.tar.gz)..."
  git add "$MEDIA_ARCHIVE"
  git commit -m "data: update media_catalog.tar.gz snapshot" || true
  git push origin main

  echo "✓ Все партии успешно отправлены в Git!"
else
  echo ""
  echo "Для пошаговой отправки в Git с рабочего сервера выполните:"
  echo ""
  echo "  # Шаг 1: Товары каталога (~300 КБ)"
  echo "  git add data/dump_products.sql.gz"
  echo "  git commit -m 'data: update products catalog dump'"
  echo "  git push origin main"
  echo ""
  if [ -f "$DUMP_EMB_V1" ]; then
    echo "  # Шаг 2: Векторы v1 DINOv2 (~300 МБ)"
    echo "  git add data/dump_embeddings_v1.sql.gz"
    echo "  git commit -m 'data: update v1 DINOv2 product embeddings dump'"
    echo "  git push origin main"
    echo ""
  fi
  if [ -f "$DUMP_EMB_V4" ]; then
    echo "  # Шаг 3: Векторы v4 SigLIP 2 (~600 МБ)"
    echo "  git add data/dump_embeddings_v4.sql.gz"
    echo "  git commit -m 'data: update v4 SigLIP 2 product embeddings dump'"
    echo "  git push origin main"
    echo ""
  fi
  echo "  # Или сразу всё по очереди автоматически:"
  echo "  ./scripts/export_production_data.sh --push"
  echo ""
fi
