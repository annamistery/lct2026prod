import base64
import io
import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from app.core.config import get_settings
from app.core.dependencies import get_detector, get_rectification, get_segmenter
from app.services.detector import DetectorService
from app.services.rectification import RectificationService
from app.services.segmenter import SegmenterService

BASE_DIR = Path(__file__).resolve().parent.parent

router = APIRouter(tags=["rectify-web"])
templates = Jinja2Templates(directory=str(BASE_DIR / "web" / "templates"))


def get_source_dir(source: str) -> Path:
    base = BASE_DIR.parent
    if source == "imports":
        d = Path("/imports/images") if Path("/imports/images").is_dir() else base / "imports" / "images"
    elif source == "tmp1":
        d = base / "tmp" / "1" / "queries"
    elif source == "tmp2":
        d = base / "tmp" / "2" / "queries"
    elif source == "catalog":
        d = base / "media" / "products"
    else:
        d = base / "imports" / "images"
    return d


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
    base = BASE_DIR.parent
    for src_id, label, path in [
        ("imports", "Каталог (imports/images)", get_source_dir("imports")),
        ("tmp1", "Тестовый пакет 1 (tmp/1/queries)", base / "tmp" / "1" / "queries"),
        ("tmp2", "Тестовый пакет 2 (tmp/2/queries)", base / "tmp" / "2" / "queries"),
        ("catalog", "Кропы этикеток (media/products)", base / "media" / "products"),
    ]:
        if path.is_dir():
            count = sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".webp", ".png"})
            sources.append({"id": src_id, "label": f"{label} ({count})", "count": count})
    return sources


@router.get("/api/rectify/files")
async def list_files(source: str = "imports", q: str = ""):
    source_dir = get_source_dir(source)
    if not source_dir.is_dir():
        return []
    allowed_exts = {".webp", ".jpg", ".jpeg", ".png"}
    files = [p.name for p in sorted(source_dir.iterdir()) if p.is_file() and p.suffix.lower() in allowed_exts]
    if q.strip():
        search = q.strip().lower()
        files = [f for f in files if search in f.lower()]
    return files[:400]


@router.get("/api/rectify/image")
async def get_image(source: str = "imports", filename: str = ""):
    source_dir = get_source_dir(source)
    file_path = (source_dir / filename).resolve()
    if source_dir not in file_path.parents or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
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

    # Get services from state or create with requested thresholds
    detector: DetectorService = request.app.state.detector
    segmenter: SegmenterService = request.app.state.segmenter

    # If thresholds differ from defaults or service missing, instantiate lightweight
    if detector is None or detector.confidence != conf_detect:
        detector = DetectorService(settings.yolo_model_path, conf_detect)

    if segmenter is None or segmenter.confidence != conf_seg:
        if settings.yolo_seg_model_path.is_file():
            segmenter = SegmenterService(settings.yolo_seg_model_path, conf_seg)

    rectifier = RectificationService(
        detector=detector,
        segmenter=segmenter,
        target_size=target_size,
    )

    with Image.open(file_path) as raw_img:
        orig_w, orig_h = raw_img.size
        orig_mode = raw_img.mode
        rgb_img = raw_img.convert("RGB")

    result = rectifier.rectify(rgb_img)
    matrix_b64 = encode_image_base64(result.rectified_image, format="WEBP", quality=95)

    return {
        "filename": filename,
        "source": source,
        "orig_width": orig_w,
        "orig_height": orig_h,
        "orig_mode": orig_mode,
        "image_url": f"/api/rectify/image?source={source}&filename={filename}",
        "bbox": result.bbox,
        "crop_bbox": result.crop_bbox,
        "seg_polygon": result.seg_polygon_orig,
        "quad_corners": result.quad_corners_orig,
        "is_fallback": result.is_fallback,
        "matrix_b64": matrix_b64,
        "matrix_size": target_size,
        "timings": {
            "bbox_ms": result.bbox_ms,
            "seg_ms": result.seg_ms,
            "warp_ms": result.warp_ms,
            "total_ms": result.total_ms,
        },
    }
