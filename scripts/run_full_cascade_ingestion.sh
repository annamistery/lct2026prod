#!/usr/bin/env bash
# Full Production Pipeline for Cascade Catalog Ingestion (v1 DINOv2 + v4 SigLIP 2)
# Generates complete 116-augmentation vector clouds for both models.

set -euo pipefail

CSV_PATH="${1:-/imports/wines_integrated_cleared.csv}"
IMAGES_DIR="${2:-/media/catalog_sources}"

echo "=========================================================================="
echo " Starting Full Cascade Ingestion Pipeline"
echo " CSV Catalog: $CSV_PATH"
echo " Images Dir:  $IMAGES_DIR"
echo " Date:        $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================================="

# Step 1: Import products into DB and build 116-vector cloud for v1 (DINOv2)
echo ""
echo ">>> [1/3] Ingesting products and building v1 vector cloud (DINOv2, 116 per item)..."
python scripts/import_from_csv.py \
    --csv "$CSV_PATH" \
    --images-dir "$IMAGES_DIR"

# Step 2: Generate 116-augmentation Letterbox 518x518 crop dataset for v4
echo ""
echo ">>> [2/3] Building v4 crop dataset (116 augmentations per item on disk)..."
python scripts/build_dataset_v4.py

# Step 3: Compute SigLIP 2 embeddings (v4) and build 116-vector cloud in product_embeddings_v4
echo ""
echo ">>> [3/3] Building v4 vector cloud (SigLIP 2, 768d, 116 per item)..."
python scripts/build_embeddings_v4.py --replace

echo ""
echo "=========================================================================="
echo " Pipeline Complete! Restarting API service..."
echo "=========================================================================="
