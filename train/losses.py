"""Supervised Contrastive Loss (SupCon, Khosla et al. 2020).

Ported from LCT2026/train/head.py for use in v3 DINOv2-base LoRA training.
"""

from __future__ import annotations

import torch


def supervised_contrastive_loss(
    features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """SupCon loss. `features` must be L2-normalized, shape (B, D)."""
    device = features.device
    batch_size = features.shape[0]
    sim = torch.matmul(features, features.T) / temperature  # (B, B)

    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()  # numerical stability

    labels = labels.view(-1, 1)
    mask_pos = torch.eq(labels, labels.T).float().to(device)
    mask_self = torch.eye(batch_size, device=device)
    mask_pos = mask_pos - mask_self  # exclude self from positives

    exp_sim = torch.exp(sim) * (1 - mask_self)  # exclude self from denominator
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = mask_pos.sum(dim=1)
    valid = pos_count > 0
    loss_per_sample = -(mask_pos * log_prob).sum(dim=1)[valid] / pos_count[valid]
    return loss_per_sample.mean()
