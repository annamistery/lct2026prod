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

        # Processor: prefer model dir, fall back to base dir
        processor_path = (
            model_path
            if (model_path / "preprocessor_config.json").exists()
            else base_model_path
        )
        self.processor = AutoImageProcessor.from_pretrained(
            str(processor_path), local_files_only=True, use_fast=False
        )
        if skip_resize:
            self.processor.do_resize = False
            self.processor.do_center_crop = False

        # Model: LoRA adapter or full weights
        if (model_path / "adapter_config.json").exists():
            if not base_model_path.is_dir():
                raise FileNotFoundError(f"SigLIP2 base model not found: {base_model_path}")
            from peft import PeftModel

            base = AutoModel.from_pretrained(str(base_model_path), local_files_only=True)
            self.model = PeftModel.from_pretrained(base, str(model_path), local_files_only=True)
        else:
            self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True)

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
            outputs = self.model(pixel_values=pixels)
            if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                features = outputs.image_embeds
            elif getattr(outputs, "pooler_output", None) is not None:
                features = outputs.pooler_output
            elif getattr(outputs, "vision_model_output", None) is not None:
                vo = outputs.vision_model_output
                features = vo.pooler_output if getattr(vo, "pooler_output", None) is not None else vo.last_hidden_state[:, 0, :]
            else:
                features = outputs.last_hidden_state[:, 0, :]
            normalized = (
                torch.nn.functional.normalize(features, dim=-1).cpu().numpy().astype(np.float32)
            )
            if normalized.shape[1] != self.expected_dimension:
                raise ValueError(
                    f"Embedding dimension is {normalized.shape[1]}, expected {self.expected_dimension}"
                )
            all_embeddings.extend(normalized.tolist())
        return all_embeddings
