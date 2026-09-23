# Руководство по Search Pipeline v4 (Google SigLIP 2 + OCR Reranker)

Версия: **v4**  
Архитектура: **Google SigLIP 2 Vision Tower (768d, LoRA fine-tuned) + pgvector (HNSW / Exact Cosine) + OCR Vintage Reranker**

Данный документ предназначен для разработчиков, интегрирующих или тестирующих поисковый пайплайн v4 в соседних ветках Git.

---

## 1. Архитектура и принцип работы v4

Пайплайн v4 решает задачу поиска вина по фотографии этикетки в условиях сильных геометрических и световых искажений камеры:

```text
[Фото пользователя]
        │
        ▼
1. Query Preparation (QueryPrepV3)
   ├── YOLO Detection: поиск BBox этикетки с сохранением естественных пропорций
   ├── YOLO Segmentation: полигональная маска контура этикетки
   ├── Очистка маски: морфологическое закрытие (close) + выпуклая оболочка (convex hull)
   └── Канонизация: Letterbox 518×518 с нулевым заполнением (черные поля, без сплющивания)
        │
        ▼
2. Извлечение эмбеддинга (SigLIP 2 Vision Tower)
   ├── Модель: google/siglip2-base-patch16-512 (Vision Tower) + LoRA адаптеры
   └── Выход: 768-мерный L2-нормализованный вектор на единичной гиперсфере
        │
        ▼
3. First-Stage Retrieval (pgvector + Vote Counting)
   ├── Поиск по таблице `product_embeddings_v4` (116 аугментаций на товар)
   ├── Отбор ближайших векторов по косинусному расстоянию: `ORDER BY embedding <=> query_vec`
   ├── Голосование (Vote Counting): подсчет количества попаданий аугментаций каждого товара в Top-N
   └── Выборка Top-K уникальных товаров каталога
        │
        ▼
4. Second-Stage Reranking (OCR Vintage & Text Check)
   ├── Извлечение текста этикетки (PaddleOCR / EasyOCR)
   ├── Парсинг года урожая (Regex: 1800–2029) из запроса и описания товара
   └── Формула финального балла:
       FinalScore = w_sim * CosineSim + w_vintage * VintageMatch + w_text * TextOverlap
        │
        ▼
[Финальная выдача: Top-1 Winner + Top-5 Candidates + Тайминги этапов]
```

---

## 2. API Эндпоинты

Все эндпоинты бэкенда доступны по префиксу `/api/` (через Apache Reverse Proxy или напрямую на порту 8030).

### 2.1. Поиск по полному фото бутылки

**`POST /api/v4/search`**

Основной эндпоинт поиска. Принимает произвольное фото бутылки с камеры смартфона, сам находит этикетку через YOLO, сегментирует, извлекает признаки и ранжирует кандидатов.

* **Content-Type:** `multipart/form-data`
* **Параметры:**
  * `image` *(обязательный, binary)*: файл изображения (JPEG, PNG, WebP до 10 МБ).
  * `k` *(опциональный, int, default=5, max=20)*: количество возвращаемых кандидатов.

**Пример запроса (cURL):**
```bash
curl -fsS -X POST http://localhost:8030/api/v4/search \
  -F "image=@/path/to/bottle_photo.jpg" \
  -F "k=5"
```

**Пример запроса (Python / requests):**
```python
import requests

with open("bottle.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8030/api/v4/search",
        files={"image": ("bottle.jpg", f, "image/jpeg")},
        data={"k": 5}
    )
data = response.json()
print("Top-1 Winner:", data["winner"]["title"], f"({data['winner']['final_score'] * 100:.1f}%)")
```

---

### 2.2. Поиск по готовому кропу этикетки

**`POST /api/v4/search-from-crop`**

Используется, когда клиент (мобильное приложение / веб-сканер) уже вырезал область этикетки на клиенте через браузерную ONNX-модель. Минует стадию детектора BBox.

* **Content-Type:** `multipart/form-data`
* **Параметры:**
  * `crop` *(обязательный, binary)*: изображение вырезанной этикетки.
  * `k` *(опциональный, int, default=5)*: число результатов.

**Пример запроса (cURL):**
```bash
curl -fsS -X POST http://localhost:8030/api/v4/search-from-crop \
  -F "crop=@/path/to/crop.webp" \
  -F "k=5"
```

---

### 2.3. Предикт Top-1 для автоматических бенчмарков

**`POST /api/v4/eval/predict`**

Ультра-быстрый эндпоинт для запуска тестовых скриптов и замера метрик Hit@1. Возвращает только строковый `slug` победителя.

* **Content-Type:** `multipart/form-data`
* **Параметры:**
  * `image` *(обязательный, binary)*: фото бутылки.

**Пример запроса (cURL):**
```bash
curl -fsS -X POST http://localhost:8030/api/v4/eval/predict \
  -F "image=@/path/to/test.jpg"
```

**Ответ:**
```json
{
  "slug": "chateau-margaux-2015"
}
```

---

### 2.4. Интерактивный веб-интерфейс

В браузере доступна страница визуального тестирования:
* **URL:** `http://<server-ip>:8030/search-v4`
* Позволяет загружать фото через drag-and-drop или выбирать из тестовых наборов (`tmp1`, `tmp2`, `imports`, `catalog`).
* Наглядно отображает 4 карточки цепочки:
  1. **Оригинал** (фото пользователя)
  2. **Seg Letterbox 518** (нормализованный кроп запроса)
  3. **Эталон Top-1** (чистый неискаженный каталог `catalog.webp`)
  4. **Совпавшая аугментация** (конкретная проекция, давшая максимальное сходство)
* Полноэкранный Lightbox при клике на любую картинку.

---

## 3. Формат JSON-ответа (`SearchResponseV4`)

```json
{
  "winner": {
    "product_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "slug": "chateau-latour-2018",
    "title": "Château Latour Grand Cru",
    "manufacturer": "Château Latour",
    "description": "Pauillac Premier Grand Cru Classé 2018",
    "image_url": "/api/media/dataset_v3/images/3fa85f64.../catalog.webp",
    "matched_aug_image": "data:image/webp;base64,UklGR...",
    "dino_similarity": 0.9234,
    "vote_count": 28,
    "vote_ratio": 0.56,
    "matched_aug_name": "perspective_1274005900",
    "ocr_vintage_match": true,
    "ocr_score": 0.89,
    "final_score": 0.912,
    "rank": 1
  },
  "results": [
    {
      "product_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      "slug": "chateau-latour-2018",
      "title": "Château Latour Grand Cru",
      "manufacturer": "Château Latour",
      "description": "Pauillac Premier Grand Cru Classé 2018",
      "image_url": "/api/media/dataset_v3/images/3fa85f64.../catalog.webp",
      "matched_aug_image": "data:image/webp;base64,...",
      "dino_similarity": 0.9234,
      "vote_count": 28,
      "vote_ratio": 0.56,
      "matched_aug_name": "perspective_1274005900",
      "ocr_vintage_match": true,
      "ocr_score": 0.89,
      "final_score": 0.912,
      "rank": 1
    },
    ... (до 5 уникальных товаров-кандидатов)
  ],
  "timings": {
    "query_prep_ms": 18.4,
    "embedding_ms": 24.1,
    "pgvector_ms": 8.7,
    "ocr_ms": 62.3,
    "total_ms": 113.5
  },
  "query_crop": "data:image/webp;base64,...",
  "bbox_crop": "data:image/webp;base64,...",
  "seg_polygon": [[120.5, 45.2], [410.2, 48.0], [405.1, 480.3], [125.0, 475.1]],
  "is_fallback": false,
  "vintage_detected": "2018"
}
```

### Поля ответа:
* `winner`: карточка Top-1 победителя (или `null`, если база пуста).
* `results`: упорядоченный список Top-K уникальных товаров-соседей.
* `dino_similarity`: косинусное сходство эмбеддингов SigLIP 2 $[0.0, 1.0]$.
* `vote_count`: сколько аугментаций данного товара попало в пул ближайших 50 векторов.
* `matched_aug_name`: тип аугментации, выигравший в голосовании (`perspective`, `cylinder_warp`, `color_jitter` и др.).
* `matched_aug_image`: Base64-изображение совпавшей аугментации (для визуального контроля искажения).
* `image_url`: ссылка на чистый эталон каталога без искажений (`catalog.webp`).
* `vintage_detected`: год урожая, автоматически извлеченный OCR из текста этикетки.
* `final_score`: взвешенный балл с учетом сходства SigLIP 2, совпадения года и пересечения текста.
* `timings`: раздельный замер времени работы каждого этапа в миллисекундах.

---

## 4. Конфигурация `.env`

Параметры v4 конфигурируются через переменные окружения:

```dotenv
# Пути к моделям (внутри Docker /models или media/models)
SIGLIP_V4_MODEL_PATH=/models/siglip2_v4_finetuned
SIGLIP_V4_BASE_MODEL_PATH=/models/siglip2-base-patch16-512

# Параметры векторизации
EMBEDDING_MODEL_V4_NAME=siglip2-v4
EMBEDDING_DIMENSION_V4=768
CANONICAL_SIZE_V4=518

# OCR Reranker
ENABLE_OCR_RERANK_V4=true
OCR_RERANK_WEIGHT_SIM=0.5
OCR_RERANK_WEIGHT_VINTAGE=0.3
OCR_RERANK_WEIGHT_TEXT=0.2

# Пулы кандидатов и голосования
V4_VOTE_POOL_SIZE=50
V4_CANDIDATE_POOL_SIZE=20
```

---

## 5. Быстрый запуск для разработчика «из коробки»

Если вы работаете в соседней ветке и хотите протестировать v4:

```bash
# 1. Забрать ветку и большие файлы через Git LFS
git checkout <ветка-с-v4>
git pull origin <ветка-с-v4>
git lfs pull

# 2. Накатить миграцию таблицы v4
docker compose run --rm migrate

# 3. Запустить проект
docker compose up -d api

# 4. Проверить статус
curl http://localhost:8030/api/ping
curl http://localhost:8030/api/ready
```
Откройте в браузере `http://localhost:8030/search-v4` и протестируйте поиск.
