from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class EmbeddingService:
    def __init__(self, model_path: Path, base_model_path: Path, expected_dimension: int):
        if not model_path.is_dir():
            raise FileNotFoundError(f"DINO model not found: {model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor_path = model_path if (model_path / "preprocessor_config.json").exists() else base_model_path
        self.processor = AutoImageProcessor.from_pretrained(str(processor_path), local_files_only=True, use_fast=False)
        if (model_path / "adapter_config.json").exists():
            if not base_model_path.is_dir():
                raise FileNotFoundError(f"DINO base model not found: {base_model_path}")
            from peft import PeftModel

            base_model = AutoModel.from_pretrained(str(base_model_path), local_files_only=True)
            self.model = PeftModel.from_pretrained(base_model, str(model_path), local_files_only=True)
        else:
            self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
        self.model = self.model.to(self.device).eval()
        self.expected_dimension = expected_dimension

    @torch.no_grad()
    def embed(self, image: Image.Image) -> list[float]:
        return self.embed_batch([image])[0]

    @torch.no_grad()
    def embed_batch(self, images: list[Image.Image], batch_size: int = 32) -> list[list[float]]:
        if not images:
            return []
        all_embeddings: list[list[float]] = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            pixels = self.processor(images=chunk, return_tensors="pt")["pixel_values"].to(self.device)
            outputs = self.model(pixels)
            features = outputs.pooler_output if getattr(outputs, "pooler_output", None) is not None else outputs.last_hidden_state[:, 0, :]
            normalized = torch.nn.functional.normalize(features, dim=-1).cpu().numpy().astype(np.float32)
            if normalized.shape[1] != self.expected_dimension:
                raise ValueError(f"Embedding dimension is {normalized.shape[1]}, expected {self.expected_dimension}")
            all_embeddings.extend(normalized.tolist())
        return all_embeddings
