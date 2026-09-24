import uuid

from pydantic import BaseModel


class TwinProduct(BaseModel):
    product_id: uuid.UUID
    title: str
    manufacturer: str
    crop_url: str
    slug: str | None = None


class TwinCluster(BaseModel):
    cluster_id: str
    manufacturer: str
    size: int
    max_similarity: float
    products: list[TwinProduct]


class TwinClustersResponse(BaseModel):
    total_clusters: int
    total_products: int
    clusters: list[TwinCluster]
