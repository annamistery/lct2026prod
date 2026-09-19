from __future__ import annotations

from collections import defaultdict
from typing import Annotated
import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.core.dependencies import get_session
from app.db.models.product import Product, ProductEmbedding
from app.schemas.twins import TwinCluster, TwinClustersResponse, TwinProduct

router = APIRouter(prefix="/twins", tags=["twins"])


@router.get("/clusters", response_model=TwinClustersResponse)
async def list_twin_clusters(
    min_similarity: Annotated[float, Query(ge=0.5, le=0.999)] = 0.88,
    limit_pairs: Annotated[int, Query(ge=10, le=5000)] = 2000,
    session: AsyncSession = Depends(get_session),
) -> TwinClustersResponse:
    """Find clusters of very similar catalog labels (twins) using pgvector cosine distance."""
    max_distance = 1.0 - min_similarity

    pe1 = aliased(ProductEmbedding)
    pe2 = aliased(ProductEmbedding)
    dist = pe1.embedding.cosine_distance(pe2.embedding)

    stmt = (
        select(
            pe1.product_id.label("p1_id"),
            pe2.product_id.label("p2_id"),
            dist.label("distance"),
        )
        .where(
            pe1.sample_type == "catalog",
            pe2.sample_type == "catalog",
            pe1.product_id < pe2.product_id,
            dist <= max_distance,
        )
        .order_by(dist.asc())
        .limit(limit_pairs)
    )

    rows = (await session.execute(stmt)).all()
    if not rows:
        return TwinClustersResponse(total_clusters=0, total_products=0, clusters=[])

    # Disjoint-Set / Union-Find for clustering connected pairs
    parent: dict[uuid.UUID, uuid.UUID] = {}

    def find(u: uuid.UUID) -> uuid.UUID:
        if parent.setdefault(u, u) != u:
            parent[u] = find(parent[u])
        return parent[u]

    def union(u: uuid.UUID, v: uuid.UUID) -> None:
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            parent[root_v] = root_u

    all_ids: set[uuid.UUID] = set()
    for p1_id, p2_id, _ in rows:
        union(p1_id, p2_id)
        all_ids.add(p1_id)
        all_ids.add(p2_id)

    # Group product IDs by their root representative
    groups: dict[uuid.UUID, list[uuid.UUID]] = defaultdict(list)
    for pid in all_ids:
        groups[find(pid)].append(pid)

    # Compute max pairwise similarity per cluster
    cluster_max_sim: dict[uuid.UUID, float] = defaultdict(float)
    for p1_id, p2_id, distance in rows:
        root = find(p1_id)
        sim = max(0.0, min(1.0, 1.0 - float(distance)))
        if sim > cluster_max_sim[root]:
            cluster_max_sim[root] = sim

    # Bulk fetch products metadata
    products_res = await session.scalars(select(Product).where(Product.id.in_(all_ids)))
    products_map = {p.id: p for p in products_res}

    clusters: list[TwinCluster] = []
    for root_id, member_ids in groups.items():
        if len(member_ids) < 2:
            continue

        cluster_products: list[TwinProduct] = []
        for pid in member_ids:
            p = products_map.get(pid)
            if p is not None:
                cluster_products.append(
                    TwinProduct(
                        product_id=p.id,
                        title=p.title,
                        manufacturer=p.manufacturer,
                        crop_url=f"/api/media/{p.label_image_path}",
                        slug=p.slug,
                    )
                )

        if len(cluster_products) < 2:
            continue

        # Sort products within cluster by title for consistent display
        cluster_products.sort(key=lambda x: x.title)

        manufacturers = [p.manufacturer for p in cluster_products if p.manufacturer]
        top_mfg = max(set(manufacturers), key=manufacturers.count) if manufacturers else "Разные производители"

        clusters.append(
            TwinCluster(
                cluster_id=f"cluster_{root_id.hex[:8]}",
                manufacturer=top_mfg,
                size=len(cluster_products),
                max_similarity=round(cluster_max_sim.get(root_id, min_similarity), 4),
                products=cluster_products,
            )
        )

    # Sort clusters: largest groups first, then highest similarity
    clusters.sort(key=lambda c: (-c.size, -c.max_similarity))

    total_products_count = sum(c.size for c in clusters)
    return TwinClustersResponse(
        total_clusters=len(clusters),
        total_products=total_products_count,
        clusters=clusters,
    )
