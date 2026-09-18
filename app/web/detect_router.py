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
    return files[:200]


@router.get("/api/detect/image/{filename:path}")
async def get_raw_import_image(filename: str):
    images_dir = get_imports_images_dir()
    file_path = (images_dir / filename).resolve()
    if images_dir not in file_path.parents or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)


@router.post("/api/detect/predict")
async def predict_import_image(request: Request, filename: Annotated[str, Form()]):
    images_dir = get_imports_images_dir()
    file_path = (images_dir / filename).resolve()
    if images_dir not in file_path.parents or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    settings = get_settings()
    detector: DetectorService = request.app.state.detector
    if detector is None:
        detector = DetectorService(settings.yolo_model_path, 0.10)  # low conf for debugging

    start_time = time.perf_counter()
    with Image.open(file_path) as img:
        orig_w, orig_h = img.size
        orig_mode = img.mode
        rgb_img = img.convert("RGB")

    # Run raw YOLO inference with low threshold to see ALL candidates
    results = detector.model.predict(source=rgb_img, conf=0.05, verbose=False)
    infer_ms = round((time.perf_counter() - start_time) * 1000, 2)

    detections = []
    for result in results:
        for box in result.boxes:
            coords = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            detections.append(
                {
                    "xtl": coords[0],
                    "ytl": coords[1],
                    "xbr": coords[2],
                    "ybr": coords[3],
                    "confidence": conf,
                    "class_id": cls_id,
                }
            )

    detections.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "filename": filename,
        "image_url": f"/api/detect/image/{filename}",
        "width": orig_w,
        "height": orig_h,
        "mode": orig_mode,
        "infer_ms": infer_ms,
        "detections": detections,
    }
