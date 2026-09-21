import base64
import io
import json
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from app.core.config import get_settings
from app.services.detector import DetectorService
from app.services.rectification import RectificationService
from app.services.segmenter import SegmenterService

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

router = APIRouter(tags=["rectify-web"])
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "app" / "web" / "templates"))


def get_source_dir(source: str) -> Path:
    if source == "tmp1":
        for cand in [PROJECT_ROOT / "tmp" / "1" / "queries", Path("/srv/app/tmp/1/queries"), Path("tmp/1/queries")]:
            if cand.is_dir():
                return cand
        return PROJECT_ROOT / "tmp" / "1" / "queries"
    elif source == "tmp2":
        for cand in [PROJECT_ROOT / "tmp" / "2" / "queries", Path("/srv/app/tmp/2/queries"), Path("tmp/2/queries")]:
            if cand.is_dir():
                return cand
        return PROJECT_ROOT / "tmp" / "2" / "queries"
    elif source == "imports":
        for cand in [Path("/imports/images"), PROJECT_ROOT / "imports" / "images", Path("imports/images")]:
            if cand.is_dir():
                return cand
        return Path("/imports/images")
    elif source == "catalog":
        for cand in [Path("/media/products"), PROJECT_ROOT / "media" / "products", Path("media/products")]:
            if cand.is_dir():
                return cand
        return Path("/media/products")
    return PROJECT_ROOT / "tmp" / "1" / "queries"


def encode_image_base64(image: Image.Image, format: str = "WEBP", quality: int = 92) -> str:
    buf = io.BytesIO()
    image.save(buf, format=format, quality=quality)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "webp" if format == "WEBP" else "jpeg"
    return f"data:image/{mime};base64,{encoded}"


@router.get("/rectify-test", response_class=HTMLResponse)
async def rectify_test_page(request: Request):
    return templates.TemplateResponse(
        "rectify_test.html",
        {
            "request": request,
            "active": "rectify_test",
        },
    )


@router.get("/api/rectify/sources")
async def list_sources():
    sources = []
    for src_id, label in [
        ("tmp1", "Тест 1 (set48 / 27 фото)"),
        ("tmp2", "Тест 2 (vina / 25 фото)"),
        ("imports", "Каталог (imports/images)"),
        ("catalog", "Кропы базы (media/products)"),
    ]:
        p = get_source_dir(src_id)
        if p.is_dir():
            if src_id == "catalog":
                nested = list(p.glob("*/label.webp"))
                count = len(nested) if nested else sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".webp", ".png"})
            else:
                count = sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".webp", ".png"})
            sources.append({"id": src_id, "label": f"{label} ({count})", "count": count})
    return sources


@router.get("/api/rectify/files")
async def list_files(source: str = "tmp1", q: str = ""):
    source_dir = get_source_dir(source)
    if not source_dir.is_dir():
        return []
    allowed_exts = {".webp", ".jpg", ".jpeg", ".png"}
    if source == "catalog":
        nested = sorted(source_dir.glob("*/label.webp"))
        if nested:
            files = [f"{p.parent.name}/{p.name}" for p in nested]
        else:
            files = [p.name for p in sorted(source_dir.iterdir()) if p.is_file() and p.suffix.lower() in allowed_exts]
    else:
        files = [p.name for p in sorted(source_dir.iterdir()) if p.is_file() and p.suffix.lower() in allowed_exts]

    if q.strip():
        search = q.strip().lower()
        files = [f for f in files if search in f.lower()]
    return files[:500]


@router.get("/api/rectify/image")
async def get_image(source: str = "tmp1", filename: str = ""):
    source_dir = get_source_dir(source)
    file_path = (source_dir / filename).resolve()
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    return FileResponse(file_path)


@router.post("/api/rectify/process")
async def process_rectification(
    request: Request,
    source: Annotated[str, Form()] = "imports",
    filename: Annotated[str, Form()] = "",
    conf_detect: Annotated[float, Form()] = 0.25,
    conf_seg: Annotated[float, Form()] = 0.25,
    target_size: Annotated[int, Form()] = 256,
):
    source_dir = get_source_dir(source)
    file_path = (source_dir / filename).resolve()
    if source_dir not in file_path.parents or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    settings = get_settings()

    # Get services from state or load with resolved paths
    detector: DetectorService = request.app.state.detector
    segmenter: SegmenterService = request.app.state.segmenter

    if detector is None:
        det_path = settings.resolved_yolo_model_path
        if det_path.is_file():
            detector = DetectorService(det_path, conf_detect)

    if segmenter is None:
        seg_path = settings.resolved_yolo_seg_model_path
        if seg_path.is_file():
            segmenter = SegmenterService(seg_path, conf_seg)

    rectifier = RectificationService(
        detector=detector,
        segmenter=segmenter,
        target_size=target_size,
    )

    try:
        with Image.open(file_path) as raw_img:
            orig_w, orig_h = raw_img.size
            orig_mode = raw_img.mode
            rgb_img = raw_img.convert("RGB")

        result = rectifier.rectify(rgb_img, conf_detect=conf_detect, conf_seg=conf_seg)
        matrix_b64 = encode_image_base64(result.rectified_image, format="WEBP", quality=95)

        return {
            "filename": filename,
            "source": source,
            "orig_width": int(orig_w),
            "orig_height": int(orig_h),
            "orig_mode": str(orig_mode),
            "image_url": f"/api/rectify/image?source={source}&filename={filename}",
            "has_detector": detector is not None,
            "has_segmenter": segmenter is not None,
            "seg_model_path": str(settings.resolved_yolo_seg_model_path),
            "bbox": [float(v) for v in result.bbox] if result.bbox else None,
            "crop_bbox": [int(v) for v in result.crop_bbox] if result.crop_bbox else None,
            "seg_polygon": [[float(p[0]), float(p[1])] for p in result.seg_polygon_orig] if result.seg_polygon_orig else None,
            "quad_corners": [[float(p[0]), float(p[1])] for p in result.quad_corners_orig] if result.quad_corners_orig else None,
            "is_fallback": bool(result.is_fallback),
            "matrix_b64": matrix_b64,
            "matrix_size": int(target_size),
            "timings": {
                "bbox_ms": float(result.bbox_ms),
                "seg_ms": float(result.seg_ms),
                "warp_ms": float(result.warp_ms),
                "total_ms": float(result.total_ms),
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@router.post("/api/rectify/export_debug")
async def export_debug_artifacts(
    request: Request,
    source: Annotated[str, Form()] = "tmp1",
    filename: Annotated[str, Form()] = "",
    conf_detect: Annotated[float, Form()] = 0.25,
    conf_seg: Annotated[float, Form()] = 0.25,
    target_size: Annotated[int, Form()] = 256,
):
    source_dir = get_source_dir(source)
    file_path = (source_dir / filename).resolve()
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    settings = get_settings()
    detector: DetectorService = request.app.state.detector
    segmenter: SegmenterService = request.app.state.segmenter

    if detector is None:
        det_path = settings.resolved_yolo_model_path
        if det_path.is_file():
            detector = DetectorService(det_path, conf_detect)

    if segmenter is None:
        seg_path = settings.resolved_yolo_seg_model_path
        if seg_path.is_file():
            segmenter = SegmenterService(seg_path, conf_seg)

    rectifier = RectificationService(detector=detector, segmenter=segmenter, target_size=target_size)

    try:
        with Image.open(file_path) as raw_img:
            orig_w, orig_h = raw_img.size
            rgb_img = raw_img.convert("RGB")

        result = rectifier.rectify(rgb_img, conf_detect=conf_detect, conf_seg=conf_seg)

        from PIL import ImageDraw
        overlay = rgb_img.copy()
        draw = ImageDraw.Draw(overlay)

        if result.bbox:
            x1, y1, x2, y2 = result.bbox
            draw.rectangle([x1, y1, x2, y2], outline="#2ed573", width=max(3, int(orig_w / 300)))

        if result.seg_polygon_orig and len(result.seg_polygon_orig) > 2:
            poly_pts = [(p[0], p[1]) for p in result.seg_polygon_orig]
            draw.polygon(poly_pts, outline="#ff4757", width=max(2, int(orig_w / 350)))

        if result.quad_corners_orig and len(result.quad_corners_orig) == 4:
            q_pts = [(p[0], p[1]) for p in result.quad_corners_orig]
            draw.polygon(q_pts, outline="#00d2d3", width=max(3, int(orig_w / 250)))
            for _idx, pt in enumerate(q_pts, 1):
                r = max(5, int(orig_w / 150))
                draw.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill="#00d2d3", outline="#000000", width=2)

        if result.crop_bbox:
            bx1, by1, bx2, by2 = result.crop_bbox
            bbox_crop_img = rgb_img.crop((bx1, by1, bx2, by2))
        else:
            bbox_crop_img = rgb_img.copy()

        # Generate Montage Collage
        card_h = 520
        ov_w = max(100, int(orig_w * (card_h / orig_h)))
        ov_resized = overlay.resize((ov_w, card_h), Image.Resampling.LANCZOS)

        bb_w = max(100, int(bbox_crop_img.width * (card_h / bbox_crop_img.height)))
        bb_resized = bbox_crop_img.resize((bb_w, card_h), Image.Resampling.LANCZOS)

        mat_resized = result.rectified_image.resize((card_h, card_h), Image.Resampling.NEAREST)

        gap = 20
        header_h = 70
        total_w = ov_w + bb_w + card_h + gap * 4
        total_h = card_h + header_h + gap

        collage = Image.new("RGB", (total_w, total_h), (20, 20, 20))
        c_draw = ImageDraw.Draw(collage)

        # ASCII only to prevent Pillow default font UnicodeEncodeError on Linux
        status_str = "FALLBACK (BBox)" if result.is_fallback else "CONTOUR UNROLL (Matrix v2)"
        c_draw.text((20, 12), f"LCT2026: {filename} | {status_str}", fill="#ffffff")
        c_draw.text((20, 36), f"BBox: {result.bbox_ms}ms | Seg: {result.seg_ms}ms | Warp: {result.warp_ms}ms | Total: {result.total_ms}ms", fill="#aaaaaa")

        x_cursor = gap
        c_draw.text((x_cursor, header_h - 18), "1. Original + BBox + Full Seg Contour", fill="#2ed573")
        collage.paste(ov_resized, (x_cursor, header_h))
        x_cursor += ov_w + gap

        c_draw.text((x_cursor, header_h - 18), "2. BBox Crop", fill="#ffffff")
        collage.paste(bb_resized, (x_cursor, header_h))
        x_cursor += bb_w + gap

        c_draw.text((x_cursor, header_h - 18), "3. Full Contour Matrix v2 (256x256)", fill="#00d2d3")
        collage.paste(mat_resized, (x_cursor, header_h))

        # Save to media/debug_exports on disk (always writable in Docker)
        stem = Path(filename).stem
        export_dir = settings.media_dir / "debug_exports" / stem

        # 1. Full YOLO Detection analysis
        det_boxes = []
        if detector is not None:
            det_res = detector.model.predict(source=rgb_img, conf=conf_detect, verbose=False)
            for r in det_res:
                for b in r.boxes:
                    c = [float(round(float(v), 1)) for v in b.xyxy[0].tolist()]
                    det_boxes.append({
                        "coords": c,
                        "conf": float(round(float(b.conf[0]), 4)),
                        "w": round(c[2] - c[0], 1),
                        "h": round(c[3] - c[1], 1),
                    })
            det_boxes.sort(key=lambda x: x["conf"], reverse=True)

        # 2. Full YOLO Segmentation analysis (on the crop)
        seg_masks = []
        crop_w_val = bbox_crop_img.width
        crop_h_val = bbox_crop_img.height
        if segmenter is not None and result.crop_bbox:
            seg_res = segmenter.model.predict(source=bbox_crop_img, conf=conf_seg, verbose=False)
            for r in seg_res:
                if r.masks is not None:
                    for idx, m_xy in enumerate(r.masks.xy):
                        pts = [[float(round(float(p[0]), 1)), float(round(float(p[1]), 1))] for p in m_xy]
                        m_conf = float(round(float(r.boxes.conf[idx]), 4)) if (r.boxes and len(r.boxes) > idx) else 0.0
                        c_area = float(abs(cv2.contourArea(np.array(pts, dtype=np.float32)))) if len(pts) >= 3 else 0.0
                        seg_masks.append({
                            "idx": idx + 1,
                            "conf": m_conf,
                            "points_count": len(pts),
                            "area_px": round(c_area, 1),
                            "area_pct": round((c_area / max(1, crop_w_val * crop_h_val)) * 100, 1),
                        })
            seg_masks.sort(key=lambda x: x["area_px"], reverse=True)

        # 3. Generate structured text report
        report_lines = [
            "=" * 80,
            f" LCT2026 ДИАГНОСТИЧЕСКИЙ ОТЧЕТ ДЕТЕКЦИИ И СЕГМЕНТАЦИИ: {filename}",
            "=" * 80,
            f"Источник: {source}",
            f"Размер оригинала: {orig_w} x {orig_h} px",
            f"Статус выравнивания: {'⚠️ FALLBACK НА BBOX' if result.is_fallback else '✓ ВЫПРЯМЛЕНИЕ (QUAD WARP)'}",
            f"Тайминги: BBox {result.bbox_ms}ms | Seg {result.seg_ms}ms | Warp {result.warp_ms}ms | Итого {result.total_ms}ms",
            "",
            "-" * 80,
            f"1. ДЕТЕКЦИЯ BBOX (YOLO Detect, порог conf={conf_detect})",
            "-" * 80,
            f"Всего найдено BBox: {len(det_boxes)}",
        ]
        for idx, b in enumerate(det_boxes, 1):
            report_lines.append(f"  [{idx}] BBox: {b['coords']} | Conf: {b['conf']} | Размер: {b['w']}x{b['h']} px")
        if result.bbox:
            report_lines.append(f"Выбран лучший BBox: {result.bbox}")
            report_lines.append(f"Кроп с 5% padding: {result.crop_bbox}")
        else:
            report_lines.append("BBox не обнаружен (fallback на весь кадр).")

        report_lines.extend([
            "",
            "-" * 80,
            f"2. СЕГМЕНТАЦИЯ НА КРОПЕ (YOLO Segment, порог conf={conf_seg})",
            "-" * 80,
            f"Размер входного кропа: {crop_w_val} x {crop_h_val} px",
            f"Всего найдено масок сегментации: {len(seg_masks)}",
        ])
        for m in seg_masks:
            report_lines.append(f"  [Маска {m['idx']}] Conf: {m['conf']} | Вершин: {m['points_count']} | Площадь: {m['area_px']} px² ({m['area_pct']}% площади кропа)")
        if result.seg_polygon_orig:
            report_lines.append(f"Выбрана лучшая маска: {len(result.seg_polygon_orig)} точек полигона.")
        else:
            report_lines.append("Маска сегментации не обнаружена.")

        report_lines.extend([
            "",
            "-" * 80,
            "3. 4 ОПОРНЫХ УГЛА ГРАНИЦ СЕГМЕНТАЦИИ (TL, TR, BR, BL)",
            "-" * 80,
        ])
        if result.quad_corners_orig and len(result.quad_corners_orig) == 4:
            q = result.quad_corners_orig
            report_lines.append(f"  1:TL (Top-Left):     ({q[0][0]}, {q[0][1]})")
            report_lines.append(f"  2:TR (Top-Right):    ({q[1][0]}, {q[1][1]})")
            report_lines.append(f"  3:BR (Bottom-Right): ({q[2][0]}, {q[2][1]})")
            report_lines.append(f"  4:BL (Bottom-Left):  ({q[3][0]}, {q[3][1]})")
            report_lines.append("  Алгоритм развертки: Coons Patch (трансфинитная интерполяция всей обводки сегментации)")
        else:
            report_lines.append("4 угла не извлечены (использован BBox fallback).")

        report_lines.append("=" * 80)
        report_text = "\n".join(report_lines)

        try:
            export_dir.mkdir(parents=True, exist_ok=True)
            collage.save(export_dir / "collage.png", format="PNG")
            overlay.save(export_dir / "01_overlay_full.png", format="PNG")
            bbox_crop_img.save(export_dir / "02_bbox_crop.png", format="PNG")
            result.rectified_image.save(export_dir / "03_matrix_256.png", format="PNG")
            (export_dir / "diagnostic_report.txt").write_text(report_text, encoding="utf-8")

            meta_payload = {
                "filename": filename,
                "source": source,
                "orig_width": int(orig_w),
                "orig_height": int(orig_h),
                "detection_boxes": det_boxes,
                "segmentation_masks": seg_masks,
                "bbox": [float(v) for v in result.bbox] if result.bbox else None,
                "crop_bbox": [int(v) for v in result.crop_bbox] if result.crop_bbox else None,
                "seg_polygon": [[float(p[0]), float(p[1])] for p in result.seg_polygon_orig] if result.seg_polygon_orig else None,
                "quad_corners": [[float(p[0]), float(p[1])] for p in result.quad_corners_orig] if result.quad_corners_orig else None,
                "is_fallback": bool(result.is_fallback),
                "timings": {
                    "bbox_ms": float(result.bbox_ms),
                    "seg_ms": float(result.seg_ms),
                    "warp_ms": float(result.warp_ms),
                    "total_ms": float(result.total_ms),
                },
            }
            (export_dir / "meta.json").write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

        collage_b64 = encode_image_base64(collage, format="PNG")

        return {
            "ok": True,
            "filename": filename,
            "server_export_dir": str(export_dir),
            "collage_b64": collage_b64,
            "report_text": report_text,
            "is_fallback": result.is_fallback,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
