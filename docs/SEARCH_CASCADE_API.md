# Руководство по Cascade Search Pipeline (v1 Coarse + v4 Refinement)

Архитектура: **Двухэтапный каскад: DINOv2-small LoRA (384d) + Scoped Google SigLIP 2 (768d) & OCR Reranker**

Данный документ описывает финальный объединенный поисковый каскад, сочетающий скорость и точность DINOv2 на уникальных этикетках с возможностями SigLIP 2 и OCR по разрешению вин-близнецов.

---

## 1. Архитектура и принцип работы каскада

```text
[Фото пользователя]
        │
        ▼
1. YOLO Detection (best_box)
   └── Извлечение BBox-кропа этикетки в оригинальном высоком разрешении
        │
        ▼
2. Этап 1: v1 Coarse Retrieval (DINOv2-small LoRA, 384d, ~15 мс)
   ├── Извлечение эмбеддинга DINOv2 (384d)
   └── Косинусный поиск в pgvector (`product_embeddings`)
        │
        ▼
3. Cascade Decision Engine (Анализ выдачи v1)
   ├── Проверка отрыва: (Sim1 - Sim2) >= 0.05 И Sim1 >= 0.65?
   ├── Проверка конфликта одного производителя/серии?
   │
   ├── [ДА: Ярко выраженный ТОП-1] ──────────────────────────────────┐
   │    └── Статус: "v1_confident"                                   │
   │    └── Мгновенный возврат ответа (время отклика ~20 мс)          │
   │                                                                 │
   └── [НЕТ: Обнаружены кандидаты-соседи]                            │
        │                                                            │
        ▼                                                            │
4. Отбор пула соседей (v1_neighbors)                                 │
   └── Список из 2–5 близких кандидатов (ID, сходство, бренд)        │
        │                                                            │
        ▼                                                            │
5. Этап 2: v4 Refinement (SigLIP 2 768d + OCR Reranker)              │
   ├── Вход: исходный оригинальный BBox кроп с этапа 1               │
   ├── QueryPrepV3: сегментация маски + Letterbox 518×518            │
   ├── SigLIP 2 Vision Tower: эмбеддинг 768d                         │
   ├── pgvector vote search: СТРОГО `WHERE product_id IN (:neighbors)`│
   └── OCR Reranker: проверка года урожая (vintage) и текста         │
        │                                                            │
        ▼                                                            │
[Финальный результат: Top-1 Winner + Top-5 + Диагностика всех этапов] ◄┘
```

---

## 2. API Эндпоинты

Все эндпоинты каскада доступны по префиксу `/api/cascade/`.

### 2.1. Полный диагностический поиск: `POST /api/cascade/search`

Принимает фото полного кадра, выполняет детекцию, проходит каскад и возвращает подробные данные по каждому этапу.

```bash
curl -X POST "http://localhost:8030/api/cascade/search" \
  -F "image=@/path/to/bottle.jpg" \
  -F "k=5"
```

Пример JSON-ответа:
```json
{
  "stage_reached": "v4_refined",
  "winner": {
    "product_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "slug": "chateau-tamagne-cabernet-2020",
    "title": "Шато Тамань Каберне 2020",
    "manufacturer": "Кубань-Вино",
    "description": "Красное сухое вино 2020 года...",
    "image_url": "/api/media/products/9b1deb4d/crop.webp",
    "dino_similarity": 0.864,
    "final_score": 0.9421,
    "vote_count": 14,
    "vote_ratio": 0.28,
    "ocr_vintage_match": true,
    "rank": 1
  },
  "decision": {
    "is_confident": false,
    "top1_similarity": 0.864,
    "top2_similarity": 0.851,
    "margin": 0.013,
    "reason": "Малый отрыв между кандидатами: 0.0130 < 0.0500 (топ-1: 0.8640, топ-2: 0.8510)"
  },
  "v1_neighbors": [
    {
      "product_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
      "slug": "chateau-tamagne-cabernet-2020",
      "title": "Шато Тамань Каберне 2020",
      "manufacturer": "Кубань-Вино",
      "dino_similarity": 0.864,
      "rank_v1": 1
    },
    {
      "product_id": "a45f9c1e-1234-4abc-9999-112233445566",
      "slug": "chateau-tamagne-merlot-2020",
      "title": "Шато Тамань Мерло 2020",
      "manufacturer": "Кубань-Вино",
      "dino_similarity": 0.851,
      "rank_v1": 2
    }
  ],
  "timings": {
    "bbox_detect_ms": 11.2,
    "v1_total_ms": 14.8,
    "decision_ms": 0.2,
    "v4_total_ms": 52.4,
    "total_ms": 78.6
  }
}
```

### 2.2. Быстрый бенчмарк-эндпоинт: `POST /api/cascade/predict`

Легковесный эндпоинт для официального тестирования и оценки точности. Возвращает только `slug` победителя, этап решения и скор уверенности.

```bash
curl -X POST "http://localhost:8030/api/cascade/predict" \
  -F "image=@/path/to/bottle.jpg"
```

Пример JSON-ответа:
```json
{
  "slug": "chateau-tamagne-cabernet-2020",
  "stage_reached": "v4_refined",
  "confidence": 0.9421
}
```

---

## 3. Веб-интерфейс каскада

Интерактивный сканер доступен в браузере по адресу:
```text
http://<ip-сервера>:8030/search-cascade
```
Интерфейс наглядно отображает:
- Баннер этапа (`Этап 1: v1 DINOv2` зеленый или `Этап 2: v4 SigLIP 2 + OCR` фиолетовый);
- Тайминги каждого компонента каскада;
- Карточки кандидатов v1 с подсветкой отобранных соседей;
- Карточки переранжирования v4;
- Итоговый результат Top-K с возможностью детального зума этикеток.
