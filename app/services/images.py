import io
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


class InvalidImage(ValueError):
    pass


@dataclass(frozen=True)
class StoredImages:
    source_path: str
    label_path: str


class ImageService:
    allowed_formats = {"JPEG", "PNG", "WEBP"}

    def __init__(self, media_dir: Path, canonical_size: int, max_bytes: int, max_pixels: int):
        self.media_dir = media_dir.resolve()
        self.canonical_size = canonical_size
        self.max_bytes = max_bytes
        self.max_pixels = max_pixels
        (self.media_dir / "products").mkdir(parents=True, exist_ok=True)

    def decode(self, contents: bytes) -> Image.Image:
        if not contents or len(contents) > self.max_bytes:
            raise InvalidImage("Image is empty or exceeds upload limit")
        try:
            image = Image.open(io.BytesIO(contents))
            if image.format not in self.allowed_formats:
                raise InvalidImage("Unsupported image format")
            if image.width * image.height > self.max_pixels:
                raise InvalidImage("Image dimensions exceed limit")
            image.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise InvalidImage("Invalid image") from exc
        return ImageOps.exif_transpose(image).convert("RGB")

    def canonical(self, image: Image.Image) -> Image.Image:
        return image.resize((self.canonical_size, self.canonical_size), Image.Resampling.LANCZOS)

    def crop(self, image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(image.width, int(box[2]) + 1)
        y2 = min(image.height, int(box[3]) + 1)
        if x2 <= x1 or y2 <= y1:
            raise InvalidImage("Detector returned an invalid bounding box")
        return self.canonical(image.crop((x1, y1, x2, y2)))

    def save_product(self, product_id: uuid.UUID, source: Image.Image, label: Image.Image) -> StoredImages:
        relative_dir = Path("products") / str(product_id)
        destination = self.media_dir / relative_dir
        destination.mkdir(parents=True, exist_ok=False)
        source_relative = relative_dir / "source.webp"
        label_relative = relative_dir / "label.webp"
        try:
            self._atomic_save(source, self.media_dir / source_relative)
            self._atomic_save(label, self.media_dir / label_relative)
            return StoredImages(source_relative.as_posix(), label_relative.as_posix())
        except Exception:
            for path in destination.iterdir():
                if path.is_file():
                    path.unlink()
            destination.rmdir()
            raise

    def resolve(self, relative_path: str) -> Path:
        candidate = (self.media_dir / relative_path).resolve()
        if candidate == self.media_dir or self.media_dir not in candidate.parents:
            raise FileNotFoundError("Media file not found")
        if not candidate.is_file():
            raise FileNotFoundError("Media file not found")
        return candidate

    def remove_product(self, product_id: uuid.UUID) -> None:
        directory = self.media_dir / "products" / str(product_id)
        if not directory.is_dir():
            return
        for path in directory.iterdir():
            if path.is_file():
                path.unlink()
        directory.rmdir()

    @staticmethod
    def _atomic_save(image: Image.Image, destination: Path) -> None:
        temporary = destination.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            image.save(temporary, format="WEBP", quality=95)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
