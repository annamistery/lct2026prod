#!/usr/bin/env python3
"""Fine-tune SigLIP 2 Vision Tower with LoRA for v4 label embeddings (768d, 512×512 input).

Downloads google/siglip2-base-patch16-512 automatically if not present locally.
Reads the v4 training manifest produced by build_dataset_v4.py.

Usage:
    python scripts/train_siglip_v4.py [--epochs 20] [--batch-size 32] [--lr 1e-4]
    python scripts/train_siglip_v4.py --full  # full fine-tune instead of LoRA
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator

# Ensure writable cache directories for container execution (same as scripts/import_from_csv.py)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("HF_HOME", "/tmp/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/huggingface")
os.environ.setdefault("TORCH_HOME", "/tmp/torch")

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from train.losses import supervised_contrastive_loss

DEFAULT_BASE_MODEL = "google/siglip2-base-patch16-512"
CANONICAL_SIZE = 512


def _resolve_paths(base_model_dir: str | None = None, output_dir: str | None = None):
    from app.core.config import get_settings
    settings = get_settings()

    docker_models = Path("/models")

    # Base model directory
    if base_model_dir:
        local_base = Path(base_model_dir)
    elif (docker_models / "siglip2-base-patch16-512").is_dir():
        local_base = docker_models / "siglip2-base-patch16-512"
    elif docker_models.is_dir() and os.access(str(docker_models), os.W_OK):
        local_base = docker_models / "siglip2-base-patch16-512"
    elif docker_models.is_dir():
        # Inside Docker when /models is mounted read-only, save downloaded base in media/models
        local_base = settings.media_dir / "models" / "siglip2-base-patch16-512"
    else:
        local_base = BASE_DIR / "models" / "siglip2-base-patch16-512"

    # Output directory
    if output_dir:
        out = Path(output_dir)
    elif docker_models.is_dir() and os.access(str(docker_models), os.W_OK):
        out = docker_models / "siglip2_v4_finetuned"
    elif docker_models.is_dir():
        # Inside Docker when /models is mounted read-only, write output to media/models
        out = settings.media_dir / "models" / "siglip2_v4_finetuned"
    else:
        out = BASE_DIR / "models" / "siglip2_v4_finetuned"

    # Manifest: prefer dataset_v4 if built, otherwise reuse existing dataset_v3
    if (settings.media_dir / "dataset_v4" / "v4_manifest.json").is_file():
        manifest = settings.media_dir / "dataset_v4" / "v4_manifest.json"
    elif (settings.media_dir / "dataset_v3" / "v3_manifest.json").is_file():
        manifest = settings.media_dir / "dataset_v3" / "v3_manifest.json"
    else:
        manifest = settings.media_dir / "dataset_v4" / "v4_manifest.json"

    return local_base, manifest, out


def ensure_base_model(model_name: str, local_base_dir: Path) -> str:
    if local_base_dir.is_dir() and (local_base_dir / "config.json").is_file():
        print(f"Using local base model: {local_base_dir}")
        return str(local_base_dir)

    print(f"Base model not found at {local_base_dir}, downloading {model_name} from HuggingFace...")
    local_base_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(local_base_dir)
    processor = AutoImageProcessor.from_pretrained(model_name)
    processor.save_pretrained(local_base_dir)
    print(f"Base model saved to {local_base_dir}")
    return str(local_base_dir)


def load_manifest(manifest_file: Path, media_dir: Path | None = None) -> list[dict]:
    if manifest_file.is_file():
        with manifest_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    # Auto-generate manifest from existing media/images/ if available
    if media_dir is not None:
        images_dir = media_dir / "images"
        if images_dir.is_dir():
            import re

            print(f"Manifest {manifest_file} not found, but found {images_dir}. Auto-generating manifest...")
            manifest = []
            for p_dir in sorted(images_dir.iterdir()):
                if not p_dir.is_dir():
                    continue
                prod_id = p_dir.name
                for f in sorted(p_dir.glob("*.webp")):
                    fname = f.name
                    if fname == "catalog.webp":
                        aug_name, aug_seed = "catalog", 0
                    else:
                        m = re.match(r"^(.+)_(\d+)\.webp$", fname)
                        aug_name = m.group(1) if m else fname
                        aug_seed = int(m.group(2)) if m else 0
                    manifest.append({
                        "image_id": f"{prod_id}_{aug_name}_{aug_seed}",
                        "group_id": prod_id,
                        "aug_name": aug_name,
                        "aug_seed": aug_seed,
                        "image_path": f"images/{prod_id}/{fname}",
                    })
            if manifest:
                manifest_file.parent.mkdir(parents=True, exist_ok=True)
                with manifest_file.open("w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
                print(f"Saved auto-manifest: {len(manifest)} images ({len(set(m['group_id'] for m in manifest))} products) to {manifest_file}")
                return manifest

    raise FileNotFoundError(
        f"v4 manifest not found: {manifest_file}\n"
        "Either place pre-built images in media/images/ or run 'python scripts/build_dataset_v4.py'."
    )


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


def _load_images(indices: list[int], manifest: list[dict], media_dir: Path) -> list[Image.Image]:
    images = []
    for idx in indices:
        img_path = media_dir / manifest[idx]["image_path"]
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
    base_model_dir: str | None = None,
    output_dir_override: str | None = None,
) -> None:
    local_base_dir, manifest_file, output_dir = _resolve_paths(base_model_dir, output_dir_override)
    from app.core.config import get_settings
    settings = get_settings()
    media_dir = settings.media_dir
    canonical_size = settings.canonical_size_v4

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_file, media_dir)
    print(f"Dataset v4: {len(manifest)} images")

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

    resolved_model = ensure_base_model(model_name, local_base_dir)
    processor = AutoImageProcessor.from_pretrained(resolved_model)

    # Images are already letterboxed to canonical_size — skip resize/crop
    processor.do_resize = False
    processor.do_center_crop = False
    processor.size = {"height": canonical_size, "width": canonical_size}
    processor.crop_size = {"height": canonical_size, "width": canonical_size}

    model = AutoModel.from_pretrained(resolved_model)

    if lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("LoRA requires peft: pip install peft") from e

        # SigLIP 2 attention projection names (Vision Tower)
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, config)
        print("LoRA enabled:")
        model.print_trainable_parameters()
    else:
        print("Full fine-tune: all parameters trainable")

    model = model.to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        except Exception as e:
            print(f"Gradient checkpointing: {e}")
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_cuda and amp_dtype == torch.float16))
    if use_cuda:
        print(f"AMP enabled: {amp_dtype} (scaler={'on' if scaler.is_enabled() else 'off'})")

    samples_per_class = max(2, batch_size // classes_per_batch)
    batch_gen = _class_balanced_batches(labels, classes_per_batch, samples_per_class, seed=seed)

    stats = {"epochs": [], "final_loss": 0.0}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            idx = next(batch_gen)
            images = _load_images(idx, manifest, media_dir)
            inputs = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            labels_t = torch.from_numpy(labels[idx]).long().to(device)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda):
                outputs = model(pixel_values=inputs)
                if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                    feats = outputs.image_embeds
                elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    feats = outputs.pooler_output
                elif hasattr(outputs, "vision_model_output") and outputs.vision_model_output is not None:
                    vo = outputs.vision_model_output
                    feats = vo.pooler_output if getattr(vo, "pooler_output", None) is not None else vo.last_hidden_state[:, 0, :]
                else:
                    feats = outputs.last_hidden_state[:, 0, :]
                feats = F.normalize(feats, dim=-1)

                loss = supervised_contrastive_loss(feats, labels_t, temperature=0.1)

            optimizer.zero_grad()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
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

    model.save_pretrained(output_dir, safe_serialization=False)
    processor.save_pretrained(output_dir)

    stats_file = output_dir / "train_stats.json"
    with stats_file.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Model saved to {output_dir}")
    print(f"Training stats: {stats_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune SigLIP 2 LoRA for v4 label embeddings")
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL, help=f"Base model name (default: {DEFAULT_BASE_MODEL})")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--classes-per-batch", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--full", action="store_true", help="Full fine-tune instead of LoRA")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-model-dir", default=None, help="Path to base SigLIP 2 model dir")
    parser.add_argument("--output-dir", default=None, help="Path to save fine-tuned model")
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
        base_model_dir=args.base_model_dir,
        output_dir_override=args.output_dir,
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
