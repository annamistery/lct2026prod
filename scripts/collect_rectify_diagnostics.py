"""Diagnostic extraction script for bottle label rectification analysis.

Designed for server-side GPU execution inside the Docker API container.
Processes all test images in tmp/1/queries, tmp/2/queries (or custom dirs),
runs detector and segmenter, extracts contours and corners, performs cylinder unrolling,
and saves full diagnostic artifacts (bbox crop, contour overlay, rectified matrix, JSON metadata).
Optionally packs everything into a single zip archive for easy download to local machine.
"""

from __future__ import annotations

import os

# Set writable config directories before importing libraries (Docker non-root support)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

import argparse
import json
import math
import shutil
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.services.detector import DetectorService
from app.services.rectification import RectificationService
from app.services.segmenter import SegmenterService


def analyze_image_geometry(
    img_path: Path,
    detector: DetectorService,
    segmenter: SegmenterService,
    rectifier: RectificationService,
    out_dir: Path | None = None,
    target_size: int = 256,
    padding_ratio: float = 0.05,
    conf_detect: float = 0.25,
    conf_seg: float = 0.25,
) -> dict:
    diag = {
        "file": str(img_path.name),
        "source": img_path.parent.parent.name,
        "path": str(img_path),
        "orig_w": 0,
        "orig_h": 0,
        "aspect_orig": 0.0,
        "bbox": None,
        "bbox_w": 0,
        "bbox_h": 0,
        "bbox_conf": 0.0,
        "crop_bbox": None,
        "crop_w": 0,
        "crop_h": 0,
        "seg_found": False,
        "seg_conf": 0.0,
        "num_poly_pts": 0,
        "poly_area": 0.0,
        "crop_area_coverage_pct": 0.0,
        "poly_bbox": None,
        "poly_w": 0,
        "poly_h": 0,
        "poly_solidity": 0.0,
        "corners": None,
        "corners_orig": None,
        "side_lengths": None,
        "side_angles_deg": None,
        "top_bottom_ratio": 0.0,
        "left_right_ratio": 0.0,
        "top_sagitta": 0.0,
        "bottom_sagitta": 0.0,
        "left_sagitta": 0.0,
        "right_sagitta": 0.0,
        "grid_jacobian_inverted": False,
        "grid_min_jacobian": 0.0,
        "grid_x_clipping_pct": 0.0,
        "grid_y_clipping_pct": 0.0,
        "status": "UNKNOWN",
        "timings_ms": {},
    }

    with Image.open(img_path) as raw:
        orig_img = raw.convert("RGB")
    orig_w, orig_h = orig_img.size
    diag["orig_w"] = orig_w
    diag["orig_h"] = orig_h
    diag["aspect_orig"] = round(orig_h / max(1, orig_w), 3)

    # 1. BBox Detection
    t_det0 = time.perf_counter()
    res_det = detector.model.predict(source=orig_img, conf=conf_detect, verbose=False)
    t_det = (time.perf_counter() - t_det0) * 1000
    diag["timings_ms"]["detector"] = round(t_det, 2)

    best_box = None
    best_conf = 0.0
    for r in res_det:
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                coords = b.xyxy[0].tolist()
                c = float(b.conf[0])
                if c > best_conf:
                    best_conf = c
                    best_box = coords

    if best_box is None:
        diag["status"] = "NO_BBOX"
        return diag

    xtl, ytl, xbr, ybr = best_box
    diag["bbox"] = [round(v, 1) for v in best_box]
    diag["bbox_conf"] = round(best_conf, 4)
    bw = xbr - xtl
    bh = ybr - ytl
    diag["bbox_w"] = round(bw, 1)
    diag["bbox_h"] = round(bh, 1)

    pad_x = int(bw * padding_ratio)
    pad_y = int(bh * padding_ratio)
    c_xtl = max(0, int(xtl) - pad_x)
    c_ytl = max(0, int(ytl) - pad_y)
    c_xbr = min(orig_w, int(xbr) + pad_x)
    c_ybr = min(orig_h, int(ybr) + pad_y)
    diag["crop_bbox"] = [c_xtl, c_ytl, c_xbr, c_ybr]

    crop_img = orig_img.crop((c_xtl, c_ytl, c_xbr, c_ybr))
    cw, ch = crop_img.size
    diag["crop_w"] = cw
    diag["crop_h"] = ch

    # 2. Segmentation on Crop
    t_seg0 = time.perf_counter()
    res_seg = segmenter.model.predict(source=crop_img, conf=conf_seg, verbose=False)
    t_seg = (time.perf_counter() - t_seg0) * 1000
    diag["timings_ms"]["segmenter"] = round(t_seg, 2)

    polygons = []
    for r in res_seg:
        if r.masks is not None:
            for idx, mask_xy in enumerate(r.masks.xy):
                if len(mask_xy) >= 4:
                    cf = float(r.boxes.conf[idx]) if (r.boxes and len(r.boxes) > idx) else 0.0
                    polygons.append((mask_xy.tolist(), cf))

    if not polygons:
        diag["status"] = "NO_SEGMENTATION"
        return diag

    best_poly_entry = max(polygons, key=lambda item: float(cv2.contourArea(np.array(item[0], dtype=np.float32))))
    poly_pts = np.array(best_poly_entry[0], dtype=np.float32)
    diag["seg_found"] = True
    diag["seg_conf"] = round(best_poly_entry[1], 4)
    diag["num_poly_pts"] = len(poly_pts)

    area = float(cv2.contourArea(poly_pts))
    diag["poly_area"] = round(area, 1)
    diag["crop_area_coverage_pct"] = round(100.0 * area / max(1.0, float(cw * ch)), 1)

    px_min, py_min = float(np.min(poly_pts[:, 0])), float(np.min(poly_pts[:, 1]))
    px_max, py_max = float(np.max(poly_pts[:, 0])), float(np.max(poly_pts[:, 1]))
    pw = px_max - px_min
    ph = py_max - py_min
    diag["poly_bbox"] = [round(px_min, 1), round(py_min, 1), round(px_max, 1), round(py_max, 1)]
    diag["poly_w"] = round(pw, 1)
    diag["poly_h"] = round(ph, 1)

    hull = cv2.convexHull(poly_pts)
    hull_area = float(cv2.contourArea(hull))
    diag["poly_solidity"] = round(area / max(1.0, hull_area), 3)

    # 3. Corner Extraction & Ordering (Current Algorithm)
    N = len(poly_pts)
    sa = 0.5 * np.sum(poly_pts[:, 0] * np.roll(poly_pts[:, 1], -1) - poly_pts[:, 1] * np.roll(poly_pts[:, 0], -1))
    if sa < 0:
        poly_pts = poly_pts[::-1]

    idx_tl = int(np.argmin(poly_pts[:, 0] + poly_pts[:, 1]))
    idx_tr = int(np.argmax(poly_pts[:, 0] - poly_pts[:, 1]))
    idx_br = int(np.argmax(poly_pts[:, 0] + poly_pts[:, 1]))
    idx_bl = int(np.argmin(poly_pts[:, 0] - poly_pts[:, 1]))

    pt_tl = poly_pts[idx_tl]
    pt_tr = poly_pts[idx_tr]
    pt_br = poly_pts[idx_br]
    pt_bl = poly_pts[idx_bl]

    diag["corners"] = {
        "TL": [round(float(pt_tl[0]), 1), round(float(pt_tl[1]), 1)],
        "TR": [round(float(pt_tr[0]), 1), round(float(pt_tr[1]), 1)],
        "BR": [round(float(pt_br[0]), 1), round(float(pt_br[1]), 1)],
        "BL": [round(float(pt_bl[0]), 1), round(float(pt_bl[1]), 1)],
    }
    diag["corners_orig"] = {
        "TL": [round(float(pt_tl[0]) + c_xtl, 1), round(float(pt_tl[1]) + c_ytl, 1)],
        "TR": [round(float(pt_tr[0]) + c_xtl, 1), round(float(pt_tr[1]) + c_ytl, 1)],
        "BR": [round(float(pt_br[0]) + c_xtl, 1), round(float(pt_br[1]) + c_ytl, 1)],
        "BL": [round(float(pt_bl[0]) + c_xtl, 1), round(float(pt_bl[1]) + c_ytl, 1)],
    }

    # 4. Geometry and Boundary Analysis
    def dist(p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    d_top = dist(pt_tl, pt_tr)
    d_bot = dist(pt_bl, pt_br)
    d_left = dist(pt_tl, pt_bl)
    d_right = dist(pt_tr, pt_br)

    diag["side_lengths"] = {
        "top": round(d_top, 1),
        "bottom": round(d_bot, 1),
        "left": round(d_left, 1),
        "right": round(d_right, 1),
    }
    diag["top_bottom_ratio"] = round(d_top / max(1.0, d_bot), 3)
    diag["left_right_ratio"] = round(d_left / max(1.0, d_right), 3)

    ang_top = math.degrees(math.atan2(pt_tr[1] - pt_tl[1], pt_tr[0] - pt_tl[0]))
    ang_bot = math.degrees(math.atan2(pt_br[1] - pt_bl[1], pt_br[0] - pt_bl[0]))
    ang_left = math.degrees(math.atan2(pt_bl[0] - pt_tl[0], pt_bl[1] - pt_tl[1]))
    ang_right = math.degrees(math.atan2(pt_br[0] - pt_tr[0], pt_br[1] - pt_tr[1]))

    diag["side_angles_deg"] = {
        "top_tilt": round(ang_top, 2),
        "bottom_tilt": round(ang_bot, 2),
        "left_tilt": round(ang_left, 2),
        "right_tilt": round(ang_right, 2),
    }

    def get_arc(i_from: int, i_to: int, step_dir: int = 1) -> np.ndarray:
        steps = ((i_to - i_from) * step_dir) % N
        return np.array([poly_pts[(i_from + step_dir * s) % N] for s in range(steps + 1)], dtype=np.float32)

    def calc_sagitta(arc_pts: np.ndarray, p_start: np.ndarray, p_end: np.ndarray) -> float:
        chord = p_end - p_start
        chord_len = np.linalg.norm(chord)
        if chord_len < 1e-4:
            return 0.0
        n = np.array([-chord[1], chord[0]]) / chord_len
        diffs = arc_pts - p_start
        dists = np.dot(diffs, n)
        idx_max = int(np.argmax(np.abs(dists)))
        return float(dists[idx_max])

    c_top_raw = get_arc(idx_tl, idx_tr, 1)
    c_right_raw = get_arc(idx_tr, idx_br, 1)
    c_bot_raw = get_arc(idx_bl, idx_br, -1)
    c_left_raw = get_arc(idx_tl, idx_bl, -1)

    diag["top_sagitta"] = round(calc_sagitta(c_top_raw, pt_tl, pt_tr), 1)
    diag["bottom_sagitta"] = round(calc_sagitta(c_bot_raw, pt_bl, pt_br), 1)
    diag["left_sagitta"] = round(calc_sagitta(c_left_raw, pt_tl, pt_bl), 1)
    diag["right_sagitta"] = round(calc_sagitta(c_right_raw, pt_tr, pt_br), 1)

    # 5. Production Rectification via new RectificationService (Column-wise Remap + Spike/Notch filter)
    t_warp0 = time.perf_counter()
    quad = np.array([pt_tl, pt_tr, pt_br, pt_bl], dtype=np.float32)
    rectified_img = rectifier.unroll_cylinder_mesh(crop_img, poly_pts.tolist(), quad=quad)
    t_warp = (time.perf_counter() - t_warp0) * 1000
    diag["timings_ms"]["warp"] = round(t_warp, 2)
    matrix_bgr = cv2.cvtColor(np.array(rectified_img), cv2.COLOR_RGB2BGR)
    crop_bgr = cv2.cvtColor(np.array(crop_img), cv2.COLOR_RGB2BGR)

    diag["status"] = "OK"

    # Save visual artifacts
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. BBox Crop (01_bbox_crop.png)
        cv2.imwrite(str(out_dir / "01_bbox_crop.png"), crop_bgr)

        # 2. Overlay with Segmentation, 4 Boundary Curves and 4 Corners (02_segmentation_contours.png)
        overlay_bgr = crop_bgr.copy()

        # Semi-transparent mask fill
        mask_layer = np.zeros_like(crop_bgr, dtype=np.uint8)
        poly_int = poly_pts.astype(np.int32)
        cv2.fillPoly(mask_layer, [poly_int], (0, 200, 255))
        cv2.addWeighted(overlay_bgr, 0.75, mask_layer, 0.25, 0, overlay_bgr)

        def draw_arc_lines(pts_arc, color, thickness=3):
            pts_i = pts_arc.astype(np.int32)
            for i in range(len(pts_i) - 1):
                cv2.line(overlay_bgr, tuple(pts_i[i]), tuple(pts_i[i + 1]), color, thickness, cv2.LINE_AA)

        draw_arc_lines(c_top_raw, (0, 255, 0), thickness=3)     # Top: Green
        draw_arc_lines(c_right_raw, (255, 0, 0), thickness=3)   # Right: Blue
        draw_arc_lines(c_bot_raw, (0, 0, 255), thickness=3)     # Bottom: Red
        draw_arc_lines(c_left_raw, (0, 255, 255), thickness=3)  # Left: Yellow

        corners_to_draw = [
            ("TL", pt_tl, (0, 255, 0)),
            ("TR", pt_tr, (255, 0, 0)),
            ("BR", pt_br, (0, 0, 255)),
            ("BL", pt_bl, (0, 255, 255)),
        ]
        for name, pt, col in corners_to_draw:
            px, py = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(overlay_bgr, (px, py), 9, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay_bgr, (px, py), 6, col, -1, cv2.LINE_AA)
            cv2.putText(
                overlay_bgr,
                name,
                (px + 8, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay_bgr,
                name,
                (px + 8, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        legend_h = 28
        cv2.rectangle(overlay_bgr, (0, 0), (cw, legend_h), (20, 20, 20), -1)
        info_txt = f"Top (G) | Right (B) | Bot (R) | Left (Y) | Jac: {diag['grid_min_jacobian']} | T/B: {diag['top_bottom_ratio']}"
        cv2.putText(overlay_bgr, info_txt, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(str(out_dir / "02_segmentation_contours.png"), overlay_bgr)

        # 3. Distorted Matrix (03_rectified_matrix.png)
        cv2.imwrite(str(out_dir / "03_rectified_matrix.png"), matrix_bgr)

        # 4. JSON Meta with detailed arc points and coordinates
        meta_data = dict(diag)
        meta_data["polygon_points"] = poly_pts.tolist()
        meta_data["contours"] = {
            "top_arc": c_top_raw.tolist(),
            "right_arc": c_right_raw.tolist(),
            "bottom_arc": c_bot_raw.tolist(),
            "left_arc": c_left_raw.tolist(),
        }
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)

    return diag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect label rectification diagnostics on server.")
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=["tmp/1/queries", "tmp/2/queries"],
        help="Directories containing query images (relative to project root or absolute).",
    )
    parser.add_argument(
        "--output-dir",
        default="media/rectify_diagnostics",
        help="Directory where diagnostic samples and reports are saved (defaults to writable media/rectify_diagnostics).",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        default=True,
        help="Automatically pack output-dir into a zip archive for easy download (default: True).",
    )
    parser.add_argument(
        "--no-zip",
        action="store_false",
        dest="zip",
        help="Do not create a zip archive.",
    )
    parser.add_argument(
        "--conf-detect",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO label detector.",
    )
    parser.add_argument(
        "--conf-seg",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO segmentation model.",
    )
    return parser.parse_args()


def resolve_output_dir(raw_dir: str) -> Path:
    p = Path(raw_dir)
    if p.is_absolute():
        return p

    # In Docker, /media is the dedicated host bind mount with write permissions
    media_root = Path("/media")
    if media_root.is_dir() and os.access(str(media_root), os.W_OK):
        parts = p.parts
        if parts and parts[0] in ("media", "tmp"):
            sub = Path(*parts[1:]) if len(parts) > 1 else Path(".")
            return media_root / sub
        return media_root / p

    return PROJECT_ROOT / p


def main():
    args = parse_args()
    settings = get_settings()

    print("=" * 80)
    print("LCT2026 SERVER RECTIFICATION DIAGNOSTIC HARNESS")
    print("=" * 80)

    model_det_path = settings.resolved_yolo_model_path
    model_seg_path = settings.resolved_yolo_seg_model_path

    if not model_det_path.is_file():
        print(f"ERROR: Detector model not found: {model_det_path}")
        sys.exit(1)
    if not model_seg_path.is_file():
        print(f"ERROR: Segmentation model not found: {model_seg_path}")
        sys.exit(1)

    print(f"Loading detector: {model_det_path}")
    detector = DetectorService(model_det_path, confidence=args.conf_detect)
    print(f"Loading segmenter: {model_seg_path}")
    segmenter = SegmenterService(model_seg_path, confidence=args.conf_seg)
    rectifier = RectificationService(detector=detector, segmenter=segmenter, target_size=256)

    all_images = []
    for d_str in args.dirs:
        d_path = Path(d_str)
        if not d_path.is_absolute():
            d_path = PROJECT_ROOT / d_path
        if d_path.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                all_images.extend(list(d_path.glob(ext)))
        else:
            print(f"WARNING: Directory not found: {d_path}")

    all_images = sorted(set(all_images))
    print(f"Found {len(all_images)} test images across target folders.\n")

    if not all_images:
        print("ERROR: No images found to process. Check --dirs arguments.")
        sys.exit(1)

    diag_base_dir = resolve_output_dir(args.output_dir)
    diag_base_dir.mkdir(parents=True, exist_ok=True)

    results = []
    report_lines = []
    report_lines.append("=" * 90)
    report_lines.append("LCT2026 СВОДНЫЙ ДИАГНОСТИЧЕСКИЙ ОТЧЕТ ВЫРАВНИВАНИЯ ЭТИКЕТОК В МАТРИЦУ")
    report_lines.append(f"Дата генерации: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Всего тестовых изображений: {len(all_images)}")
    report_lines.append("=" * 90)
    report_lines.append("")

    for idx, img_path in enumerate(all_images, 1):
        try:
            rel_path = img_path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_path = img_path.name
        folder_prefix = img_path.parent.parent.name
        sample_out_dir = diag_base_dir / f"{folder_prefix}_{img_path.stem}"

        diag = analyze_image_geometry(
            img_path,
            detector,
            segmenter,
            rectifier,
            out_dir=sample_out_dir,
            conf_detect=args.conf_detect,
            conf_seg=args.conf_seg,
        )
        results.append(diag)

        status_icon = "✓" if diag["status"] == "OK" and not diag["grid_jacobian_inverted"] else "⚠"
        print(f"[{idx:02d}/{len(all_images):02d}] {status_icon} {rel_path} | BBox: {diag['bbox_w']}x{diag['bbox_h']} | Poly: {diag['num_poly_pts']} pts ({diag['crop_area_coverage_pct']}%) | Sagitta: top={diag['top_sagitta']} bot={diag['bottom_sagitta']} | Jac: {diag['grid_min_jacobian']}")

        report_lines.append("-" * 90)
        report_lines.append(f"ФАЙЛ [{idx:02d}/{len(all_images):02d}]: {rel_path} (Статус: {diag['status']})")
        report_lines.append("-" * 90)
        report_lines.append(f"  Оригинал:      {diag['orig_w']} x {diag['orig_h']} px (соотношение H/W: {diag['aspect_orig']})")
        if diag["bbox"]:
            report_lines.append(f"  BBox детекция: [{diag['bbox'][0]}, {diag['bbox'][1]}, {diag['bbox'][2]}, {diag['bbox'][3]}] (conf: {diag['bbox_conf']})")
            report_lines.append(f"                 размер: {diag['bbox_w']} x {diag['bbox_h']} px, кроп с padding: {diag['crop_bbox']} ({diag['crop_w']}x{diag['crop_h']} px)")
        else:
            report_lines.append("  BBox детекция: НЕ НАЙДЕН")

        if diag["seg_found"]:
            report_lines.append(f"  Сегментация:   найдена (conf: {diag['seg_conf']}), точек в контуре: {diag['num_poly_pts']}")
            report_lines.append(f"                 площадь: {diag['poly_area']} px² ({diag['crop_area_coverage_pct']}% площади кропа)")
            report_lines.append(f"                 BBox маски: {diag['poly_bbox']} ({diag['poly_w']}x{diag['poly_h']} px), solidity: {diag['poly_solidity']}")
            if diag["corners"]:
                c = diag["corners"]
                report_lines.append(f"  Опорные углы:  TL=({c['TL'][0]}, {c['TL'][1]}), TR=({c['TR'][0]}, {c['TR'][1]})")
                report_lines.append(f"                 BL=({c['BL'][0]}, {c['BL'][1]}), BR=({c['BR'][0]}, {c['BR'][1]})")
            if diag["side_lengths"]:
                sl = diag["side_lengths"]
                report_lines.append(f"  Длины сторон:  верх={sl['top']} px, низ={sl['bottom']} px (отношение T/B: {diag['top_bottom_ratio']})")
                report_lines.append(f"                 лево={sl['left']} px, право={sl['right']} px (отношение L/R: {diag['left_right_ratio']})")
            if diag["side_angles_deg"]:
                sa = diag["side_angles_deg"]
                report_lines.append(f"  Наклоны (град): верх={sa['top_tilt']}°, низ={sa['bottom_tilt']}°, лево={sa['left_tilt']}°, право={sa['right_tilt']}°")
            report_lines.append(f"  Стрела прогиба: верх(top)={diag['top_sagitta']} px, низ(bot)={diag['bottom_sagitta']} px, лево={diag['left_sagitta']} px, право={diag['right_sagitta']} px")
            report_lines.append(f"  Сетка Coons:   мин Якобиан={diag['grid_min_jacobian']} (инверсия/складки: {'ДА - СЛОМАНА!' if diag['grid_jacobian_inverted'] else 'нет'}), клиппинг X={diag['grid_x_clipping_pct']}%, Y={diag['grid_y_clipping_pct']}%")
        else:
            report_lines.append("  Сегментация:   НЕ НАЙДЕНА (fallback)")
        report_lines.append("")

    valid_results = [r for r in results if r["status"] == "OK"]
    report_lines.append("=" * 90)
    report_lines.append("СТАТИСТИЧЕСКАЯ СВОДКА И АНОМАЛИИ ВЫРАВНИВАНИЯ")
    report_lines.append("=" * 90)
    report_lines.append(f"Всего файлов: {len(results)}")
    report_lines.append(f"Успешный BBox: {sum(1 for r in results if r['bbox'] is not None)} / {len(results)}")
    report_lines.append(f"Успешная сегментация: {sum(1 for r in results if r['seg_found'])} / {len(results)}")
    report_lines.append(f"Складки/вырождение сетки Coons (Jacobian <= 0): {sum(1 for r in valid_results if r['grid_jacobian_inverted'])} / {len(valid_results)}")

    if valid_results:
        avg_top_sagitta = sum(abs(r["top_sagitta"]) for r in valid_results) / len(valid_results)
        avg_bot_sagitta = sum(abs(r["bottom_sagitta"]) for r in valid_results) / len(valid_results)
        max_tb_ratio = max(r["top_bottom_ratio"] for r in valid_results)
        min_tb_ratio = min(r["top_bottom_ratio"] for r in valid_results)
        avg_tb_ratio = sum(r["top_bottom_ratio"] for r in valid_results) / len(valid_results)
        avg_lr_ratio = sum(r["left_right_ratio"] for r in valid_results) / len(valid_results)

        report_lines.append(f"Средняя кривизна верхней дуги (|Sagitta|): {avg_top_sagitta:.1f} px")
        report_lines.append(f"Средняя кривизна нижней дуги (|Sagitta|):  {avg_bot_sagitta:.1f} px")
        report_lines.append(f"Отношение ширин Верх/Низ (Top/Bottom): мин={min_tb_ratio:.2f}, макс={max_tb_ratio:.2f}, среднее={avg_tb_ratio:.2f}")
        report_lines.append(f"Отношение высот Лево/Право (Left/Right): среднее={avg_lr_ratio:.2f}")

    report_lines.append("")
    report_lines.append("ТОП-10 АНОМАЛИЙ (перекосы, вырождения сетки, экстремальная кривизна):")
    sorted_anomalies = sorted(
        valid_results,
        key=lambda r: (
            1 if r["grid_jacobian_inverted"] else 0,
            abs(1.0 - r["top_bottom_ratio"]) + abs(1.0 - r["left_right_ratio"]),
            abs(r["top_sagitta"]) + abs(r["bottom_sagitta"]),
        ),
        reverse=True,
    )
    for a in sorted_anomalies[:10]:
        report_lines.append(
            f"  - {a['file']} ({a['source']}): T/B ratio={a['top_bottom_ratio']}, L/R ratio={a['left_right_ratio']}, "
            f"Sagitta(top={a['top_sagitta']}, bot={a['bottom_sagitta']}), MinJac={a['grid_min_jacobian']}"
        )

    report_content = "\n".join(report_lines)

    out_txt = diag_base_dir / "rectify_diagnostics_report.txt"
    out_json = diag_base_dir / "rectify_diagnostics_summary.json"

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report_content)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nArtifacts saved in: {diag_base_dir}")
    print(f"Report saved to: {out_txt}")
    print(f"JSON summary saved to: {out_json}")

    if args.zip:
        zip_path = diag_base_dir.parent / f"{diag_base_dir.name}.zip"
        print(f"Packaging archive: {zip_path} ...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(diag_base_dir):
                for file in files:
                    full_p = Path(root) / file
                    rel_p = full_p.relative_to(diag_base_dir)
                    zf.write(full_p, arcname=str(rel_p))
        print(f"Zip archive ready: {zip_path}")

    print("=" * 80)


if __name__ == "__main__":
    main()
