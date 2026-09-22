"""Perspective-frame rectification evaluation.

Runs the full RectificationService on a set of query images, saves diagnostic
artifacts for each file, and produces a JSON/TXT report.

Artifacts per image:
    01_overlay_full.png  - original image with bbox, segmentation polygon and frame
    02_bbox_crop.png     - detector crop
    03_frame_overlay.png - crop with the perspective frame drawn
    04_perspective_256.png
    frame.json           - source frame corners, aspect ratio, method, timings
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.services.detector import DetectorService
from app.services.rectification import RectificationService
from app.services.segmenter import SegmenterService


def ensure_writable_dir(raw_dir: str) -> Path:
    p = Path(raw_dir)
    if p.is_absolute():
        return p

    media_root = Path("/media")
    if media_root.is_dir() and os.access(str(media_root), os.W_OK):
        parts = p.parts
        if parts and parts[0] in ("media", "tmp"):
            sub = Path(*parts[1:]) if len(parts) > 1 else Path(".")
            return media_root / sub
        return media_root / p

    return PROJECT_ROOT / p


def draw_overlay(
    image: Image.Image,
    bbox: tuple[float, float, float, float] | None,
    polygon: list[list[float]] | None,
    frame: list[list[float]] | None,
) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    if bbox:
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="#2ed573", width=max(3, int(image.width / 300)))
    if polygon and len(polygon) > 2:
        pts = [(p[0], p[1]) for p in polygon]
        draw.polygon(pts, outline="#ff4757", width=max(2, int(image.width / 350)))
    if frame and len(frame) == 4:
        q = [(p[0], p[1]) for p in frame]
        draw.polygon(q, outline="#00d2d3", width=max(3, int(image.width / 250)))
        r = max(5, int(image.width / 150))
        for idx, pt in enumerate(q, 1):
            draw.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill="#00d2d3")
            draw.text((pt[0] + r + 2, pt[1] - r - 2), str(idx), fill="#00d2d3")
    return overlay


def process_image(
    img_path: Path,
    detector: DetectorService,
    segmenter: SegmenterService,
    rectifier: RectificationService,
    out_dir: Path,
    conf_detect: float = 0.25,
    conf_seg: float = 0.25,
) -> dict[str, Any]:
    diag: dict[str, Any] = {
        "file": img_path.name,
        "source": img_path.parent.parent.name,
        "status": "PENDING",
    }

    with Image.open(img_path) as raw:
        full_image = raw.convert("RGB")
    diag["orig_w"], diag["orig_h"] = full_image.size

    t0 = time.perf_counter()
    result = rectifier.rectify(full_image, conf_detect=conf_detect, conf_seg=conf_seg)
    diag["timings_ms"] = {
        "bbox_ms": result.bbox_ms,
        "seg_ms": result.seg_ms,
        "warp_ms": result.warp_ms,
        "total_ms": result.total_ms,
    }
    diag["method"] = result.method
    diag["is_fallback"] = result.is_fallback

    out_dir.mkdir(parents=True, exist_ok=True)

    overlay = draw_overlay(full_image, result.bbox, result.seg_polygon_orig, result.quad_corners_orig)
    overlay.save(out_dir / "01_overlay_full.png", format="PNG")

    if result.crop_bbox:
        x1, y1, x2, y2 = result.crop_bbox
        crop_img = full_image.crop((x1, y1, x2, y2))
        crop_img.save(out_dir / "02_bbox_crop.png", format="PNG")

        frame_crop = (
            [[p[0] - x1, p[1] - y1] for p in result.quad_corners_orig]
            if result.quad_corners_orig
            else None
        )
        frame_overlay = draw_overlay(crop_img, None, None, frame_crop)
        frame_overlay.save(out_dir / "03_frame_overlay.png", format="PNG")

    result.rectified_image.save(out_dir / "04_perspective_256.png", format="PNG")

    diag["frame"] = result.quad_corners_orig
    diag["bbox"] = result.bbox
    diag["crop_bbox"] = result.crop_bbox
    diag["status"] = "OK"

    (out_dir / "frame.json").write_text(
        json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return diag


def main() -> None:
    parser = argparse.ArgumentParser(description="Perspective-frame rectification evaluation")
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=["tmp/1/queries", "tmp/2/queries"],
        help="Directories containing query images",
    )
    parser.add_argument(
        "--output-dir",
        default="media/rectify_perspective",
        help="Directory for report and artifacts",
    )
    parser.add_argument(
        "--conf-detect",
        type=float,
        default=0.25,
        help="YOLO detection confidence threshold",
    )
    parser.add_argument(
        "--conf-seg",
        type=float,
        default=0.25,
        help="YOLO segmentation confidence threshold",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Do not create a zip archive",
    )
    args = parser.parse_args()

    settings = get_settings()
    det_path = settings.resolved_yolo_model_path
    seg_path = settings.resolved_yolo_seg_model_path

    if not det_path.is_file():
        print(f"ERROR: detector model not found: {det_path}")
        sys.exit(1)
    if not seg_path.is_file():
        print(f"ERROR: segmenter model not found: {seg_path}")
        sys.exit(1)

    print("Loading detector...")
    detector = DetectorService(det_path, confidence=args.conf_detect)
    print("Loading segmenter...")
    segmenter = SegmenterService(seg_path, confidence=args.conf_seg)
    print("Building rectifier...")
    rectifier = RectificationService(
        detector=detector,
        segmenter=segmenter,
        target_size=settings.canonical_size,
    )

    all_images: list[Path] = []
    for d_str in args.dirs:
        d_path = Path(d_str)
        if not d_path.is_absolute():
            d_path = PROJECT_ROOT / d_path
        if d_path.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                all_images.extend(d_path.glob(ext))
        else:
            print(f"WARNING: directory not found: {d_path}")

    all_images = sorted(set(all_images))
    print(f"Found {len(all_images)} images to process\n")
    if not all_images:
        sys.exit(1)

    out_root = ensure_writable_dir(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    report_lines: list[str] = []
    report_lines.append("=" * 80)
    report_lines.append("LCT2026 PERSPECTIVE-FRAME RECTIFICATION EVALUATION REPORT")
    report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Images: {len(all_images)}")
    report_lines.append("=" * 80)
    report_lines.append("")

    for idx, img_path in enumerate(all_images, 1):
        try:
            rel_path = img_path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_path = img_path.name
        sample_dir = out_root / f"{img_path.parent.parent.name}_{img_path.stem}"

        try:
            diag = process_image(
                img_path,
                detector,
                segmenter,
                rectifier,
                sample_dir,
                conf_detect=args.conf_detect,
                conf_seg=args.conf_seg,
            )
            results.append(diag)
            print(
                f"[{idx:02d}/{len(all_images):02d}] OK  {rel_path} "
                f"| method={diag['method']} "
                f"| fallback={diag['is_fallback']}"
            )
        except Exception as exc:
            diag = {
                "file": img_path.name,
                "source": img_path.parent.parent.name,
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            }
            results.append(diag)
            print(f"[{idx:02d}/{len(all_images):02d}] ERR {rel_path}: {exc}")

        report_lines.append(
            f"[{idx:02d}/{len(all_images):02d}] {diag['file']}  status={diag['status']}"
        )
        if diag["status"] == "OK":
            report_lines.append(
                f"    method={diag['method']}  fallback={diag['is_fallback']}  "
                f"timings={diag['timings_ms']}"
            )
            if diag.get("frame"):
                report_lines.append(f"    frame={diag['frame']}")
        elif "error" in diag:
            report_lines.append(f"    ERROR: {diag['error']}")
        report_lines.append("")

    ok = [r for r in results if r["status"] == "OK"]
    fallbacks = sum(1 for r in ok if r["is_fallback"])
    report_lines.append("=" * 80)
    report_lines.append("SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(f"Total images: {len(results)}")
    report_lines.append(f"OK:           {len(ok)}")
    report_lines.append(f"Errors:       {len(results) - len(ok)}")
    report_lines.append(f"Fallbacks:    {fallbacks}")
    report_lines.append(f"Output dir:   {out_root}")

    report_text = "\n".join(report_lines)
    (out_root / "perspective_report.txt").write_text(report_text, encoding="utf-8")
    (out_root / "perspective_report.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nArtifacts saved to: {out_root}")
    print(f"Report: {out_root / 'perspective_report.txt'}")

    if not args.no_zip:
        zip_path = out_root.parent / f"{out_root.name}.zip"
        print(f"Packing archive: {zip_path} ...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(out_root):
                for file in files:
                    full_p = Path(root) / file
                    rel_p = full_p.relative_to(out_root)
                    zf.write(full_p, arcname=str(rel_p))
        print(f"Archive ready: {zip_path}")


if __name__ == "__main__":
    main()
