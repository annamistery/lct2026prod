import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from app.core.config import get_settings
from app.services.detector import DetectorService

BASE_DIR = Path(__file__).resolve().parent.parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "web" / "templates"))

router = APIRouter(include_in_schema=False)


def get_imports_images_dir() -> Path:
    if Path("/imports/images").is_dir():
        return Path("/imports/images")
    return BASE_DIR / "imports" / "images"


def prepare_phone_frame(
    img: Image.Image,
    target_ratio: float = 3 / 4,
    bg_color: tuple[int, int, int] = (15, 15, 17),  # dark surface matching the dark theme #0f0f11
) -> tuple[Image.Image, int, int]:
    """Strip transparency and pad sides to match phone camera aspect ratio."""
    # 1. Remove transparency by pasting onto dark background
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        alpha = img.split()[-1]
        canvas = Image.new("RGB", img.size, bg_color)
        canvas.paste(img, mask=alpha)
        base = canvas
    else:
        base = img.convert("RGB")

    # 2. Add side padding to reach camera 3:4 aspect ratio
    w, h = base.size
    target_w = max(w, int(h * target_ratio))
    framed = Image.new("RGB", (target_w, h), bg_color)
    offset_x = (target_w - w) // 2
    framed.paste(base, (offset_x, 0))

    return framed, offset_x, 0


@router.get("/detect-test", response_class=HTMLResponse)
async def detect_test_page(request: Request):
    return templates.TemplateResponse(
        "detect_test.html",
        {
            "request": request,
            "active": "detect_test",
        },
    )


@router.get("/api/detect/files")
async def list_import_files(q: Annotated[str, Query()] = ""):
    images_dir = get_imports_images_dir()
    if not images_dir.is_dir():
        return []
    allowed_exts = {".webp", ".jpg", ".jpeg", ".png"}
    files = [p.name for p in sorted(images_dir.iterdir()) if p.is_file() and p.suffix.lower() in allowed_exts]
    if q.strip():
        search = q.strip().lower()
        files = [f for f in files if search in f.lower()]
    return files[:300]


@router.get("/api/detect/image/{filename:path}")
async def get_raw_import_image(filename: str):
    images_dir = get_imports_images_dir()
    file_path = (images_dir / filename).resolve()
    if images_dir not in file_path.parents or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)


@router.post("/api/detect/predict")
async def predict_import_image(
    request: Request,
    filename: Annotated[str, Form()],
    conf: Annotated[float, Form()] = 0.25,
    mode: Annotated[str, Form()] = "raw",  # 'raw' or 'phone_frame'
):
    images_dir = get_imports_images_dir()
    file_path = (images_dir / filename).resolve()
    if images_dir not in file_path.parents or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    settings = get_settings()
    detector: DetectorService = request.app.state.detector
    if detector is None:
        detector = DetectorService(settings.yolo_model_path, conf)

    start_time = time.perf_counter()
    with Image.open(file_path) as raw_img:
        orig_w, orig_h = raw_img.size
        orig_mode = raw_img.mode

        if mode == "phone_frame":
            # Strip transparency and add side padding (camera frame 3:4 ratio on dark background)
            prepared, offset_x, offset_y = prepare_phone_frame(raw_img, target_ratio=0.75, bg_color=(15, 15, 17))
            feed_img = prepared
        else:
            feed_img = raw_img.convert("RGB")
            offset_x = 0
            offset_y = 0

    feed_w, feed_h = feed_img.size

    # Run YOLO on the prepared image
    results = detector.model.predict(source=feed_img, conf=conf, verbose=False)
    infer_ms = round((time.perf_counter() - start_time) * 1000, 2)

    detections = []
    for result in results:
        for box in result.boxes:
            coords = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            cls_id = int(box.cls[0])

            # Frame coordinates (on feed_img)
            f_xtl, f_ytl, f_xbr, f_ybr = coords[0], coords[1], coords[2], coords[3]

            # Original coordinates (mapped back to bottle)
            o_xtl = max(0.0, f_xtl - offset_x)
            o_ytl = max(0.0, f_ytl - offset_y)
            o_xbr = min(float(orig_w), f_xbr - offset_x)
            o_ybr = min(float(orig_h), f_ybr - offset_y)

            detections.append(
                {
                    "xtl": round(f_xtl, 1),
                    "ytl": round(f_ytl, 1),
                    "xbr": round(f_xbr, 1),
                    "ybr": round(f_ybr, 1),
                    "orig_xtl": round(o_xtl, 1),
                    "orig_ytl": round(o_ytl, 1),
                    "orig_xbr": round(o_xbr, 1),
                    "orig_ybr": round(o_ybr, 1),
                    "width": round(f_xbr - f_xtl, 1),
                    "height": round(f_ybr - f_ytl, 1),
                    "confidence": round(confidence, 4),
                    "class_id": cls_id,
                }
            )

    detections.sort(key=lambda x: x["confidence"], reverse=True)

    # If phone_frame, save temporary preview image to /tmp or return data
    preview_url = f"/api/detect/image/{filename}"
    if mode == "phone_frame":
        import io
        import base64

        buf = io.BytesIO()
        feed_img.save(buf, format="JPEG", quality=90)
        preview_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "filename": filename,
        "image_url": preview_url,
        "width": feed_w,
        "height": feed_h,
        "orig_width": orig_w,
        "orig_height": orig_h,
        "mode": orig_mode,
        "prepared_mode": mode,
        "offset_x": offset_x,
        "infer_ms": infer_ms,
        "threshold": conf,
        "detections": detections,
    }
