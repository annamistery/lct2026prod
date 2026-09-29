"""Audit of the label crop: which YOLO box the detector picks on every photo of a folder.

For each photo writes all detected label boxes, the chosen one and whether the chosen box covers the
image centre; draws a preview (all boxes grey, the chosen one green, the image centre red) so wrong
side-label crops are easy to spot.

    docker compose exec api python scripts/audit_label_crops.py --photos /tmp/photos --out /tmp/crop_audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from app.core.config import get_settings
from app.services.detector import DetectorService

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    settings = get_settings()
    detector = DetectorService(settings.resolved_yolo_model_path, settings.yolo_confidence)
    (args.out / "preview").mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(p for p in args.photos.iterdir() if p.suffix.lower() in IMAGE_EXT):
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        boxes = detector.boxes(image)
        chosen = detector.best_box(image)
        cx, cy = image.width / 2, image.height / 2
        covers = bool(chosen and chosen[0] <= cx <= chosen[2] and chosen[1] <= cy <= chosen[3])
        rows.append({"photo": path.name, "size": image.size, "boxes": boxes, "chosen": chosen, "covers_centre": covers})

        preview = image.copy()
        preview.thumbnail((900, 900))
        scale = preview.width / image.width
        draw = ImageDraw.Draw(preview)
        for x1, y1, x2, y2, conf in boxes:
            draw.rectangle([x1 * scale, y1 * scale, x2 * scale, y2 * scale], outline=(160, 160, 160), width=2)
            draw.text((x1 * scale + 3, y1 * scale + 3), f"{conf:.2f}", fill=(255, 255, 0))
        if chosen:
            draw.rectangle([c * scale for c in chosen], outline=(0, 220, 0), width=5)
        draw.ellipse([cx * scale - 8, cy * scale - 8, cx * scale + 8, cy * scale + 8], fill=(255, 0, 0))
        preview.save(args.out / "preview" / f"{path.stem}.jpg", quality=85)

    (args.out / "crop_audit.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    bad = [row["photo"] for row in rows if row["chosen"] and not row["covers_centre"]]
    print(f"photos {len(rows)}, no box {sum(1 for r in rows if not r['chosen'])}, chosen box misses the centre {len(bad)}")
    for name in bad:
        print("  ", name)


if __name__ == "__main__":
    main()
