import io

import pytest
from PIL import Image

from app.services.images import ImageService, InvalidImage


def image_bytes(size=(40, 20), format="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", size, "red").save(buffer, format=format)
    return buffer.getvalue()


def test_decode_crop_and_canonical(tmp_path):
    service = ImageService(tmp_path, 256, 100_000, 1_000_000)
    image = service.decode(image_bytes())
    crop = service.crop(image, (-10, 0, 20, 20))
    assert crop.size == (256, 256)


def test_rejects_unsupported_and_oversized(tmp_path):
    service = ImageService(tmp_path, 256, 10, 1_000_000)
    with pytest.raises(InvalidImage):
        service.decode(image_bytes())


def test_media_path_is_confined(tmp_path):
    service = ImageService(tmp_path, 256, 100_000, 1_000_000)
    with pytest.raises(FileNotFoundError):
        service.resolve("../secret")
