"""SigLIP 2 embedding service for search pipeline v4.

Functionally mirrors EmbeddingService (DINOv2) but loads a SigLIP 2 Vision Tower
via AutoModel / AutoProcessor from HuggingFace (local files only at runtime).
Supports LoRA adapters through peft.PeftModel.

Output: CLS-token (pooler_output if available, else last_hidden_state[:, 0, :]),
        L2-normalised to unit sphere, shape (768,).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class SigLIP2EmbeddingService:
    """Inference wrapper for a (possibly LoRA-adapted) SigLIP 2 Vision Tower."""

    def __init__(
        self,
        model_path: Path,
        base_model_path: Path,
        expected_dimension: int = 768,
        skip_resize: bool = False,
    ) -> None:
        if not model_path.is_dir():
            raise FileNotFoundError(f"SigLIP2 model not found: {model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.skip_resize = skip_resize
        self.expected_dimension = expected_dimension

        # Processor: prefer model dir, then base dir, fallback to HF hub
        processor_path = None
        for p in (model_path, base_model_path):
            if (p / "preprocessor_config.json").exists():
                processor_path = str(p)
                break
        if processor_path is None:
            processor_path = "google/siglip2-base-patch16-512"

        self.processor = AutoImageProcessor.from_pretrained(
            processor_path,
            local_files_only=(processor_path != "google/siglip2-base-patch16-512"),
            use_fast=False,
        )
        if skip_resize:
            self.processor.do_resize = False
            self.processor.do_center_crop = False

        # Model: LoRA adapter or full weights
        if (model_path / "adapter_config.json").exists():
            from peft import PeftModel

            has_local_base = (
                base_model_path.is_dir()
                and (
                    (base_model_path / "model.safetensors").exists()
                    or (base_model_path / "pytorch_model.bin").exists()
                )
            )
            base_src = str(base_model_path) if has_local_base else "google/siglip2-base-patch16-512"
            base = AutoModel.from_pretrained(base_src, local_files_only=has_local_base)
            self.model = PeftModel.from_pretrained(base, str(model_path), local_files_only=True)
        else:
            has_local_model = (
                model_path.is_dir()
                and (
                    (model_path / "model.safetensors").exists()
                    or (model_path / "pytorch_model.bin").exists()
                )
            )
            model_src = str(model_path) if has_local_model else "google/siglip2-base-patch16-512"
            self.model = AutoModel.from_pretrained(model_src, local_files_only=has_local_model)

        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def embed(self, image: Image.Image) -> list[float]:
        """Embed a single PIL image → normalised 768d vector."""
        return self.embed_batch([image])[0]

    @torch.no_grad()
    def embed_batch(self, images: list[Image.Image], batch_size: int = 32) -> list[list[float]]:
        """Embed a list of PIL images in mini-batches → list of normalised 768d vectors."""
        if not images:
            return []
        all_embeddings: list[list[float]] = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            pixels = self.processor(images=chunk, return_tensors="pt")["pixel_values"].to(self.device)
            if hasattr(self.model, "get_image_features"):
                features = self.model.get_image_features(pixel_values=pixels)
            elif hasattr(self.model, "vision_model"):
                vo = self.model.vision_model(pixel_values=pixels)
                features = vo.pooler_output if getattr(vo, "pooler_output", None) is not None else vo.last_hidden_state[:, 0, :]
            else:
                outputs = self.model(pixel_values=pixels)
                features = getattr(outputs, "image_embeds", None) or getattr(outputs, "pooler_output", None) or outputs.last_hidden_state[:, 0, :]
            normalized = (
                torch.nn.functional.normalize(features, dim=-1).cpu().numpy().astype(np.float32)
            )
            if normalized.shape[1] != self.expected_dimension:
                raise ValueError(
                    f"Embedding dimension is {normalized.shape[1]}, expected {self.expected_dimension}"
                )
            all_embeddings.extend(normalized.tolist())
        return all_embeddings
