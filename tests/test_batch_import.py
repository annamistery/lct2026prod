import json

import pytest
from PIL import Image

from app.services.batch_import import BatchImportService
from app.services.images import ImageService


def service(tmp_path):
    staging = tmp_path / "imports"
    staging.mkdir()
    images = ImageService(tmp_path / "media", 256, 100_000, 1_000_000)
    return BatchImportService(staging, 100, 100_000, images, None, None), staging


def test_loads_json_and_csv_manifests(tmp_path):
    importer, staging = service(tmp_path)
    batch = staging / "customer-001"
    (batch / "images").mkdir(parents=True)
    Image.new("RGB", (10, 10)).save(batch / "images" / "one.jpg")
    (batch / "manifest.json").write_text(json.dumps({"products": [{"title": "One", "manufacturer": "Maker", "image_path": "images/one.jpg"}]}), encoding="utf-8")
    (batch / "manifest.csv").write_text("title,manufacturer,description,image_path\nOne,Maker,Text,images/one.jpg\n", encoding="utf-8")
    assert importer.load_manifest("customer-001", "manifest.json")[0].title == "One"
    assert importer.load_manifest("customer-001", "manifest.csv")[0].description == "Text"


def test_rejects_manifest_image_traversal(tmp_path):
    importer, staging = service(tmp_path)
    batch = staging / "customer-001"
    batch.mkdir()
    outside = staging / "outside.jpg"
    Image.new("RGB", (10, 10)).save(outside)
    (batch / "manifest.json").write_text(json.dumps([{"title": "One", "manufacturer": "Maker", "image_path": "../outside.jpg"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="outside the batch"):
        importer.load_manifest("customer-001", "manifest.json")
