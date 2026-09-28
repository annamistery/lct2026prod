#!/usr/bin/env bash
# Prints the current public HTTPS links of the testing entry (docker compose --profile public up -d).
# The tunnel address changes after reconnects: run this again and resend the link.
set -euo pipefail
cd "$(dirname "$0")/.."

url=""
for _ in $(seq 1 30); do
  url=$(docker compose --profile public --profile cloudflare logs tunnel tunnel-cloudflare 2>/dev/null \
    | grep -oE 'https://[a-z0-9-]+\.(lhr\.life|trycloudflare\.com)' | tail -1 || true)
  [ -n "$url" ] && break
  sleep 2
done
if [ -z "$url" ]; then
  echo "Tunnel address not found. Is it running? docker compose --profile public up -d" >&2
  exit 1
fi

echo "Сканер:            $url/mobi/"
echo "Каскадный поиск:   $url/search-cascade"
echo "API (slug):        curl -X POST $url/api/cascade/predict -F \"image=@photo.jpg\""
echo "Swagger:           $url/api/docs"
