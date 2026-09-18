import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from ultralytics import YOLO

from app.core.config import get_settings
from app.services.detector import DetectorService

BASE_DIR = Path(__file__).resolve().parent.parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "web" / "templates"))

router = APIRouter(include_in_schema=False)
_seg_model: YOLO | None = None


def get_imports_images_dir() -> Path:
    if Path("/imports/images").is_dir():
        return Path("/imports/images")
    return BASE_DIR / "imports" / "images"


def get_seg_model() -> YOLO | None:
    global _seg_model
    if _seg_model is None:
        seg_path = Path("/models/yolo_seg.pt") if Path("/models/yolo_seg.pt").is_file() else BASE_DIR / "models" / "yolo_seg.pt"
        if seg_path.is_file():
            _seg_model = YOLO(str(seg_path))
    return _seg_model


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
):
    images_dir = get_imports_images_dir()
    file_path = (images_dir / filename).resolve()
    if images_dir not in file_path.parents or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    settings = get_settings()
    detector: DetectorService = request.app.state.detector
    if detector is None:
        detector = DetectorService(settings.yolo_model_path, conf)

    # 1. Read pure original without any artificial padding/framing
    start_time = time.perf_counter()
    with Image.open(file_path) as raw_img:
        orig_w, orig_h = raw_img.size
        orig_mode = raw_img.mode
        rgb_img = raw_img.convert("RGB")

    # 2. Run BBox Detect model (pure original)
    results = detector.model.predict(source=rgb_img, conf=conf, verbose=False)
    infer_ms = round((time.perf_counter() - start_time) * 1000, 2)

    detections = []
    for result in results:
        for box in result.boxes:
            coords = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            cls_id = int(box.cls[0])
            detections.append(
                {
                    "xtl": round(coords[0], 1),
                    "ytl": round(coords[1], 1),
                    "xbr": round(coords[2], 1),
                    "ybr": round(coords[3], 1),
                    "width": round(coords[2] - coords[0], 1),
                    "height": round(coords[3] - coords[1], 1),
                    "confidence": round(confidence, 4),
                    "class_id": cls_id,
                }
            )
    detections.sort(key=lambda x: x["confidence"], reverse=True)

    # 3. Run Segmentation model
    seg_model = get_seg_model()
    segmentations = []
    if seg_model is not None:
        seg_results = seg_model.predict(source=rgb_img, conf=conf, verbose=False)
        for s_res in seg_results:
            if s_res.masks is not None:
                for idx, mask_poly in enumerate(s_res.masks.xy):
                    poly_points = [[round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in mask_poly]
                    s_conf = float(s_res.boxes.conf[idx]) if (s_res.boxes and len(s_res.boxes) > idx) else 0.0
                    segmentations.append({"points": poly_points, "confidence": round(s_conf, 4)})

    return {
        "filename": filename,
        "image_url": f"/api/detect/image/{filename}",
        "width": orig_w,
        "height": orig_h,
        "mode": orig_mode,
        "infer_ms": infer_ms,
        "threshold": conf,
        "detections": detections,
        "segmentations": segmentations,
    }
