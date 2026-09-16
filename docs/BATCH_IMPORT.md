# Пакетная загрузка датасета

## Архив

Поддерживаются ZIP и TAR/TAR.GZ. В корне архива должен лежать `manifest.json` или `manifest.csv`; изображения могут находиться в подпапках.

JSON:

```json
{
  "products": [
    {
      "title": "Название",
      "manufacturer": "Производитель",
      "description": "Описание",
      "image_path": "images/product-001.jpg"
    }
  ]
}
```

CSV:

```csv
title,manufacturer,description,image_path
Название,Производитель,Описание,images/product-001.jpg
```

Допустимые синонимы: `name` для `title`, `producer` для `manufacturer`, `image` или `filename` для `image_path`.

## Подготовка на сервере

Загрузить архив в:

```text
~/lct2026prod/imports/inbox/
```

Затем:

```bash
cd ~/lct2026prod
./scripts/prepare_dataset_import.sh imports/inbox/customer.zip customer-001
```

Скрипт безопасно распакует данные в:

```text
imports/staging/customer-001/
```

Абсолютные пути, `..`, symlink, hardlink и device entries блокируются.

## Запуск

Для JSON:

```bash
curl -fsS -X POST http://127.0.0.1:8030/api/imports \
  -H 'Content-Type: application/json' \
  -d '{"batch_id":"customer-001","manifest_name":"manifest.json"}'
```

Для CSV заменить имя на `manifest.csv`. API возвращает `job_id` и сразу продолжает импорт в фоне.

## Статус

```bash
curl -fsS http://127.0.0.1:8030/api/imports/<job_id>
```

Поля `total_items`, `completed_items`, `failed_items` показывают прогресс. `items` содержит последние 100 результатов и ошибки. Полный импорт может содержать успешные и ошибочные строки; итоговый статус `completed` означает, что весь manifest обработан.

Каждый успешный элемент создаёт товар, серверный YOLO-кроп, DINOv2 embedding и запись `product_embeddings`. Исходное изображение и кроп сохраняются в `media/`; архив и staging в Git не попадают.
