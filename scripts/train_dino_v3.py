#!/usr/bin/env python3
"""Fine-tune DINOv2-base with LoRA for v3 label embeddings (768d, 518×518 input).

Downloads facebook/dinov2-base automatically if not present locally.
Reads the v3 training manifest produced by build_dataset_v3.py.

Usage:
    python scripts/train_dino_v3.py [--epochs 20] [--batch-size 32] [--lr 1e-4]
    python scripts/train_dino_v3.py --full  # full fine-tune instead of LoRA
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, Dinov2Config

from train.losses import supervised_contrastive_loss

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BASE_MODEL = "facebook/dinov2-base"
LOCAL_BASE_MODEL_DIR = BASE_DIR / "models" / "dinov2-base"
MANIFEST_FILE = BASE_DIR / "media" / "dataset_v3" / "v3_manifest.json"
OUTPUT_DIR = BASE_DIR / "models" / "dinov2_label_finetuned_v3"
CANONICAL_SIZE = 518


def ensure_base_model(model_name: str) -> str:
    """Check for local model, download from HuggingFace if missing. Return resolved path."""
    if LOCAL_BASE_MODEL_DIR.is_dir() and (LOCAL_BASE_MODEL_DIR / "config.json").is_file():
        print(f"Using local base model: {LOCAL_BASE_MODEL_DIR}")
        return str(LOCAL_BASE_MODEL_DIR)

    print(f"Base model not found at {LOCAL_BASE_MODEL_DIR}, downloading {model_name} from HuggingFace...")
    LOCAL_BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(LOCAL_BASE_MODEL_DIR)
    processor = AutoImageProcessor.from_pretrained(model_name)
    processor.save_pretrained(LOCAL_BASE_MODEL_DIR)
    print(f"Base model saved to {LOCAL_BASE_MODEL_DIR}")
    return str(LOCAL_BASE_MODEL_DIR)


def load_manifest() -> list[dict]:
    if not MANIFEST_FILE.is_file():
        raise FileNotFoundError(
            f"v3 manifest not found: {MANIFEST_FILE}\n"
            f"Run 'python scripts/build_dataset_v3.py' first."
        )
    with MANIFEST_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def _class_balanced_batches(
    labels: np.ndarray, classes_per_batch: int, samples_per_class: int, seed: int = 0,
) -> Iterator[list[int]]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_class[int(label)].append(idx)
    class_ids = list(by_class.keys())
    rng = random.Random(seed)

    while True:
        chosen = rng.sample(class_ids, min(classes_per_batch, len(class_ids)))
        batch: list[int] = []
        for c in chosen:
            pool = by_class[c]
            if len(pool) >= samples_per_class:
                batch.extend(rng.sample(pool, samples_per_class))
            else:
                batch.extend(rng.choices(pool, k=samples_per_class))
        yield batch


def _load_images(indices: list[int], manifest: list[dict]) -> list[Image.Image]:
    images = []
    for idx in indices:
        img_path = BASE_DIR / manifest[idx]["image_path"]
        images.append(Image.open(img_path).convert("RGB"))
    return images


def finetune(
    model_name: str,
    epochs: int,
    batch_size: int,
    classes_per_batch: int,
    lr: float,
    weight_decay: float,
    steps_per_epoch: int,
    lora: bool,
    lora_r: int,
    lora_alpha: int,
    seed: int,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    print(f"Dataset v3: {len(manifest)} images")

    product_to_class: dict[str, int] = {}
    for entry in manifest:
        product = entry["group_id"]
        if product not in product_to_class:
            product_to_class[product] = len(product_to_class)

    labels = np.array([product_to_class[e["group_id"]] for e in manifest], dtype=np.int64)
    num_classes = len(product_to_class)
    print(f"Classes (products): {num_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    resolved_model = ensure_base_model(model_name)
    processor = AutoImageProcessor.from_pretrained(resolved_model)

    # Override processor to use 518×518 without center crop
    processor.size = {"height": CANONICAL_SIZE, "width": CANONICAL_SIZE}
    processor.crop_size = {"height": CANONICAL_SIZE, "width": CANONICAL_SIZE}
    processor.do_center_crop = False

    model = AutoModel.from_pretrained(resolved_model)

    if lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("LoRA requires peft: pip install peft") from e

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query", "key", "value", "dense"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, config)
        print("LoRA enabled:")
        model.print_trainable_parameters()
    else:
        print("Full fine-tune: all parameters trainable")

    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    samples_per_class = max(2, batch_size // classes_per_batch)
    batch_gen = _class_balanced_batches(labels, classes_per_batch, samples_per_class, seed=seed)

    stats = {"epochs": [], "final_loss": 0.0}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            idx = next(batch_gen)
            images = _load_images(idx, manifest)
            inputs = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            labels_t = torch.from_numpy(labels[idx]).long().to(device)

            outputs = model(inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                feats = outputs.pooler_output
            else:
                feats = outputs.last_hidden_state[:, 0, :]
            feats = F.normalize(feats, dim=-1)

            loss = supervised_contrastive_loss(feats, labels_t, temperature=0.1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / steps_per_epoch
        elapsed = time.time() - t0
        print(f"epoch {epoch}/{epochs} loss={avg_loss:.4f} time={elapsed:.1f}s")
        stats["epochs"].append({"epoch": epoch, "loss": round(avg_loss, 6)})
        scheduler.step()

    stats["final_loss"] = stats["epochs"][-1]["loss"] if stats["epochs"] else 0.0

    model.save_pretrained(OUTPUT_DIR, safe_serialization=False)
    processor.save_pretrained(OUTPUT_DIR)

    stats_file = OUTPUT_DIR / "train_stats.json"
    with stats_file.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Model saved to {OUTPUT_DIR}")
    print(f"Training stats: {stats_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DINOv2-base LoRA for v3 label embeddings")
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL, help=f"Base model name (default: {DEFAULT_BASE_MODEL})")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--classes-per-batch", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--full", action="store_true", help="Full fine-tune instead of LoRA")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    finetune(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        classes_per_batch=args.classes_per_batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        steps_per_epoch=args.steps_per_epoch,
        lora=not args.full,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
