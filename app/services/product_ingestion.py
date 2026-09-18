import asyncio
import uuid

from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Product, ProductEmbedding
from app.db.repositories.products import ProductRepository
from app.pipelines.search.v1.pipeline import SearchPipelineV1
from app.services.augment import generate_augmented_cloud
from app.services.detector import DetectorService
from app.services.images import ImageService


class ProductIngestionService:
    def __init__(self, images: ImageService, detector: DetectorService, pipeline: SearchPipelineV1, embedding_model_name: str):
        self.images = images
        self.detector = detector
        self.pipeline = pipeline
        self.embedding_model_name = embedding_model_name

    async def create(self, session: AsyncSession, title: str, manufacturer: str, description: str, source: Image.Image, slug: str | None = None) -> Product:
        clean_title = title.strip()
        clean_manufacturer = manufacturer.strip()
        if not clean_title or not clean_manufacturer:
            raise ValueError("Title and manufacturer must not be blank")

        # 1. Detect label with YOLO; fallback to canonical full image if not detected
        async with self.pipeline.gpu_semaphore:
            box = await asyncio.to_thread(self.detector.best_box, source)

        if box is None:
            # Fallback: squash full bottle image to canonical 256x256 square (same as legacy v5 add_product)
            label = self.images.canonical(source)
        else:
            label = self.images.crop(source, box)

        # 2. Generate 116 augmentation crops (1 catalog original + 23 augs * 5 variants)
        augmented_items = await asyncio.to_thread(generate_augmented_cloud, label, 5)
        augmented_images = [img for _, img in augmented_items]

        # 3. Extract 116 embeddings in GPU batches
        async with self.pipeline.gpu_semaphore:
            embeddings_list = await asyncio.to_thread(self.pipeline.embeddings.embed_batch, augmented_images, 32)

        # 4. Persist product and all 116 embeddings atomically
        product_id = uuid.uuid4()
        stored = None
        try:
            stored = await asyncio.to_thread(self.images.save_product, product_id, source, label)
            product = Product(
                id=product_id,
                slug=slug.strip() if slug and slug.strip() else None,
                title=clean_title,
                manufacturer=clean_manufacturer,
                description=description.strip(),
                source_image_path=stored.source_path,
                label_image_path=stored.label_path,
            )

            product_embeddings = []
            for (aug_name, _), embedding in zip(augmented_items, embeddings_list, strict=True):
                sample_type = "catalog" if aug_name == "catalog" else "augmented"
                product_embeddings.append(
                    ProductEmbedding(
                        image_path=stored.label_path,
                        sample_type=sample_type,
                        embedding=embedding,
                        embedding_model=self.embedding_model_name,
                    )
                )

            repo = ProductRepository(session)
            for emb in product_embeddings:
                repo.add(product, emb)

            await session.commit()
            await session.refresh(product)
            return product
        except Exception:
            await session.rollback()
            if stored is not None:
                await asyncio.to_thread(self.images.remove_product, product_id)
            raise
