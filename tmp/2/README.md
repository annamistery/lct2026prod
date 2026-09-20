# Owner eval pack 2 — vina_2026-09-18 (ручная разметка)

Полевые фото из `data/vina_2026-09-18`, только кейсы с подтверждённым slug
в каталоге (после ручной правки `mapping.json`).

| | |
|--|--:|
| positives (unique slug) | 25 |
| excluded (нет в базе / очищены) | 32 |
| duplicate slugs dropped | 0 |
| slug нет в локальной БД | 1 |

`search_was_wrong=true` — исходный `/v1/eval/predict` дал не тот top-1;
в mapping уже правильный `expected_slug` / `product_url`.

## Прогон

```bash
API__PORT=8080 PYTHONPATH=src uv run python scripts/run_api.py \
  --config config/app.yaml --no-index

cd data/test_dataset/owner_eval/2
rm -f predictions.jsonl
./participant_test.sh \
  --images-dir ./queries \
  --manifest ./queries.tsv \
  --endpoint 'http://127.0.0.1:8080/v1/eval/predict' \
  --output ./predictions.jsonl
```
