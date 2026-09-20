# Owner eval pack (set48 → формат организатора)

Пакет из **27 каталожных** магазинных фото `data/test_dataset/full/`
(без негативов) в раскладке `docs/main_eval`: непрозрачные имена файлов,
манифест `queries.tsv`.

Ответы организатору **не входят** в `queries.tsv`. Наши метки — только в
`mapping.json` (не отправлять вместе с `predictions.jsonl`).

## Состав

| путь | что внутри |
|------|------------|
| `queries/` | 27 фото, имя = первые 8 символов SHA-256 + исходное расширение |
| `queries.tsv` | `query_id` + `image_path`, порядок positives из `data/eval/ground_truth_set48.json` |
| `mapping.json` | query_id → исходный файл, `expected_slug` / `external_id` |
| `predictions.golden.json` | эталон выхода при 100% hit@1 (`latency_ms: 0`) |
| `predictions.golden.jsonl` | то же построчно — так пишет `participant_test.sh` |
| `participant_test.sh` | копия скрипта организатора |

## Прогон

Нужен запущенный API с прогретым hybrid-оркестратором (`orchestrator: true` в
`/api/health`). Локальная копия скрипта использует `--max-time 60` (CPU);
оригинал в `docs/main_eval` — 10 с (ориентир под GPU).

```bash
# из корня репозитория
API__PORT=8080 PYTHONPATH=src uv run python scripts/run_api.py \
  --config config/app.yaml --no-index

cd data/test_dataset/owner_eval
rm -f predictions.jsonl
./participant_test.sh \
  --images-dir ./queries \
  --manifest ./queries.tsv \
  --endpoint 'http://127.0.0.1:8080/v1/eval/predict' \
  --output ./predictions.jsonl
```

Ожидаемый ответ сервиса: `{"slug":"agora-muskat-chernyj"}` (или массив
`[{"slug":"..."}]`). Для всех 27 кадров есть slug в каталоге.

Эталон при идеальном Top-1 — `predictions.golden.json`. Скрипт организатора
пишет то же самое в `predictions.jsonl` (по объекту на строку); копия этого
формата есть в `predictions.golden.jsonl`.

Сверка hit@1 с нашими метками (не отдавать организатору):

```bash
PYTHONPATH=src uv run python - <<'PY'
import json
from pathlib import Path
mapping = {c["query_id"]: c["expected_slug"]
           for c in json.loads(Path("mapping.json").read_text())["cases"]}
preds = [json.loads(l) for l in Path("predictions.jsonl").read_text().splitlines() if l.strip()]
ok = sum(1 for p in preds if p.get("predicted_slug") == mapping.get(p["query_id"]))
print(f"hit@1 {ok}/{len(preds)} = {ok/len(preds):.1%}")
PY
```
