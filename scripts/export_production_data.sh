#!/usr/bin/env bash
# ==============================================================================
# Экспорт базы данных PostgreSQL (со всеми эмбеддингами) и эталонных кропов
# для воспроизводимости проекта «из коробки» на стороне заказчика.
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="$ROOT_DIR/data"
mkdir -p "$DATA_DIR"

echo "=== [1/4] Проверка окружения и Docker Compose ==="
if ! command -v docker >/dev/null 2>&1; then
  echo "ОШИБКА: docker не найден" >&2
  exit 1
fi

if ! docker compose ps --services --filter "status=running" | grep -q "^db$"; then
  echo "ОШИБКА: Сервис базы данных 'db' не запущен. Запустите: docker compose up -d db" >&2
  exit 1
fi

# Считываем переменные из .env при наличии
POSTGRES_USER="lct2026"
POSTGRES_DB="lct2026"
if [ -f .env ]; then
  val_user=$(grep -E '^POSTGRES_USER=' .env | cut -d '=' -f2- | tr -d ' "\r' || true)
  val_db=$(grep -E '^POSTGRES_DB=' .env | cut -d '=' -f2- | tr -d ' "\r' || true)
  [ -n "$val_user" ] && POSTGRES_USER="$val_user"
  [ -n "$val_db" ] && POSTGRES_DB="$val_db"
fi

echo "Используется база: '$POSTGRES_DB' (пользователь: '$POSTGRES_USER')"

echo "=== [2/4] Экспорт дампа PostgreSQL (products + product_embeddings) ==="
DUMP_FILE="$DATA_DIR/catalog_dump.sql.gz"
docker compose exec -T db pg_dump \
  -U "$POSTGRES_USER" \
  --clean \
  --if-exists \
  --no-owner \
  --no-privileges \
  "$POSTGRES_DB" | gzip -9 > "$DUMP_FILE"

DUMP_SIZE=$(du -h "$DUMP_FILE" | cut -f1)
echo "✓ Дамп базы успешно создан: $DUMP_FILE ($DUMP_SIZE)"

echo "=== [3/4] Упаковка эталонных кропов медиа-хранилища ==="
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
echo "✓ Архив кропов успешно создан: $MEDIA_ARCHIVE ($MEDIA_SIZE)"

echo "=== [4/4] Проверка Git LFS ==="
if command -v git-lfs >/dev/null 2>&1; then
  git lfs install >/dev/null 2>&1 || true
  git lfs track "data/*.sql.gz" >/dev/null 2>&1 || true
  git lfs track "data/*.tar.gz" >/dev/null 2>&1 || true
  echo "✓ Git LFS настроен для отслеживания архивов в data/"
else
  echo "ПРЕДУПРЕЖДЕНИЕ: git-lfs не установлен на хосте. Установите: sudo apt install git-lfs && git lfs install"
fi

echo ""
echo "========================================================================"
echo " Экспорт успешно завершен!"
echo " Файлы готовы к отправке в Git:"
echo "   - $DUMP_FILE ($DUMP_SIZE)"
echo "   - $MEDIA_ARCHIVE ($MEDIA_SIZE)"
echo ""
echo " Чтобы отправить данные в репозиторий, выполните:"
echo "   git add .gitattributes data/"
echo "   git commit -m 'Export production catalog database and media snapshot'"
echo "   git push origin main"
echo "========================================================================"
