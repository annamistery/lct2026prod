import asyncio
import uuid

from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import Product, ProductEmbedding, ProductEmbeddingV2, ProductEmbeddingV4
from app.db.repositories.products import ProductRepository
from app.pipelines.search.v1.pipeline import SearchPipelineV1
from app.services.augment import generate_augmented_cloud_v3
from app.services.detector import DetectorService
from app.services.images import ImageService
from app.services.query_prep_v3 import letterbox_pil
from app.services.siglip_embeddings import SigLIP2EmbeddingService


class ProductIngestionService:
    """Adds one wine to the catalog with the same reference geometry as scripts/ingest_cascade_catalog.py.

    YOLO crop (full resolution) -> letterbox 518 -> 116-image augmentation cloud -> DINOv2 (v1) and
    SigLIP 2 (v4) vectors. The cascade compares queries with exactly these references, so a wine added
    here (single upload or batch import) is recognised the same way as the pre-built catalog.
    """

    def __init__(
        self,
        images: ImageService,
        detector: DetectorService,
        pipeline: SearchPipelineV1,
        embedding_model_name: str,
        v4_embeddings: SigLIP2EmbeddingService | None = None,
    ):
        self.images = images
        self.detector = detector
        self.pipeline = pipeline
        self.embedding_model_name = embedding_model_name
        self.v4_embeddings = v4_embeddings

    async def create(self, session: AsyncSession, title: str, manufacturer: str, description: str, source: Image.Image, slug: str | None = None) -> Product:
        clean_title = title.strip()
        clean_manufacturer = manufacturer.strip()
        if not clean_title or not clean_manufacturer:
            raise ValueError("Title and manufacturer must not be blank")
        slug = slug.strip() if slug and slug.strip() else None
        if slug and await ProductRepository(session).get_by_slug(slug) is not None:
            raise ValueError(f"Product with slug '{slug}' already exists")
        settings = get_settings()

        # 1. Label crop with YOLO (full resolution); the whole photo when no label is detected
        async with self.pipeline.gpu_semaphore:
            box = await asyncio.to_thread(self.detector.best_box, source)
        raw_crop = self._crop(source, box)

        # 2. Canonical letterbox 518x518 and the 116-image augmentation cloud (1 catalog + 23 augs x 5)
        label = letterbox_pil(raw_crop, settings.canonical_size_v4)
        augmented_items = await asyncio.to_thread(generate_augmented_cloud_v3, label, 5, 42)
        augmented_images = [image for _, _, image in augmented_items]

        # 3. DINOv2 (v1) and SigLIP 2 (v4) vectors in GPU batches
        async with self.pipeline.gpu_semaphore:
            v1_vectors = await asyncio.to_thread(self.pipeline.embeddings.embed_batch, augmented_images, 32)
        v4_vectors = None
        if self.v4_embeddings is not None:
            async with self.pipeline.gpu_semaphore:
                v4_vectors = await asyncio.to_thread(self.v4_embeddings.embed_batch, augmented_images, 32)

        # 4. Persist the product and all vectors atomically
        product_id = uuid.uuid4()
        stored = None
        try:
            stored = await asyncio.to_thread(self.images.save_product, product_id, source, label)
            product = Product(
                id=product_id,
                slug=slug,
                title=clean_title,
                manufacturer=clean_manufacturer,
                description=description.strip(),
                source_image_path=stored.source_path,
                label_image_path=stored.label_path,
            )

            repo = ProductRepository(session)
            for (aug_name, _, _), embedding in zip(augmented_items, v1_vectors, strict=True):
                sample_type = "catalog" if aug_name == "catalog" else "augmented"
                repo.add(product, ProductEmbedding(image_path=stored.label_path, sample_type=sample_type, embedding=embedding, embedding_model=self.embedding_model_name))
                repo.add_embedding_v2(
                    ProductEmbeddingV2(product_id=product_id, image_path=stored.label_path, sample_type=sample_type, embedding=embedding, embedding_model=settings.embedding_model_v2_name)
                )
            if v4_vectors is not None:
                session.add_all(
                    ProductEmbeddingV4(
                        product_id=product_id,
                        image_path=stored.label_path,
                        sample_type="catalog" if aug_name == "catalog" else "augmented",
                        aug_name=aug_name,
                        aug_seed=variant_seed,
                        embedding=embedding,
                        embedding_model=settings.embedding_model_v4_name,
                    )
                    for (aug_name, variant_seed, _), embedding in zip(augmented_items, v4_vectors, strict=True)
                )

            await session.commit()
            await session.refresh(product)
            return product
        except Exception:
            await session.rollback()
            if stored is not None:
                await asyncio.to_thread(self.images.remove_product, product_id)
            raise

    @staticmethod
    def _crop(image: Image.Image, box: tuple[float, float, float, float] | None) -> Image.Image:
        if box is None:
            return image
        x1, y1, x2, y2 = box
        left, top = max(0, int(x1)), max(0, int(y1))
        right, bottom = min(image.width, int(x2)), min(image.height, int(y2))
        if right <= left or bottom <= top:
            return image
        return image.crop((left, top, right, bottom))
