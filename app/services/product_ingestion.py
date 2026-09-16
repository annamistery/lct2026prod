import asyncio
import uuid

from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Product, ProductEmbedding
from app.db.repositories.products import ProductRepository
from app.pipelines.search.v1.pipeline import SearchPipelineV1
from app.services.detector import DetectorService
from app.services.images import ImageService, InvalidImage


class ProductIngestionService:
    def __init__(self, images: ImageService, detector: DetectorService, pipeline: SearchPipelineV1, embedding_model_name: str):
        self.images = images
        self.detector = detector
        self.pipeline = pipeline
        self.embedding_model_name = embedding_model_name

    async def create(self, session: AsyncSession, title: str, manufacturer: str, description: str, source: Image.Image) -> Product:
        clean_title = title.strip()
        clean_manufacturer = manufacturer.strip()
        if not clean_title or not clean_manufacturer:
            raise ValueError("Title and manufacturer must not be blank")
        async with self.pipeline.gpu_semaphore:
            box = await asyncio.to_thread(self.detector.best_box, source)
        if box is None:
            raise InvalidImage("Label was not detected")
        label = self.images.crop(source, box)
        async with self.pipeline.gpu_semaphore:
            embedding = await asyncio.to_thread(self.pipeline.embeddings.embed, label)
        product_id = uuid.uuid4()
        stored = None
        try:
            stored = await asyncio.to_thread(self.images.save_product, product_id, source, label)
            product = Product(id=product_id, title=clean_title, manufacturer=clean_manufacturer, description=description.strip(), source_image_path=stored.source_path, label_image_path=stored.label_path)
            product_embedding = ProductEmbedding(image_path=stored.label_path, sample_type="catalog", embedding=embedding, embedding_model=self.embedding_model_name)
            ProductRepository(session).add(product, product_embedding)
            await session.commit()
            await session.refresh(product)
            return product
        except Exception:
            await session.rollback()
            if stored is not None:
                await asyncio.to_thread(self.images.remove_product, product_id)
            raise
