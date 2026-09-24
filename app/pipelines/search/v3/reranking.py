"""SIFT reranker v3: compares query against the regenerated augmented crop
that produced the closest embedding, not the flat catalog reference."""

from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from app.services.augment import _AUG_REGISTRY
from app.services.query_prep_v3 import letterbox_pil


@dataclass(frozen=True)
class SiftResultV3:
    score: float
    inliers: int
    inlier_ratio: float


class SiftRerankerV3:
    def __init__(self, max_features: int = 1000, canonical_size: int = 518):
        self.max_features = max_features
        self.canonical_size = canonical_size

    def regenerate_augment(
        self,
        label_image: Image.Image,
        aug_name: str,
        aug_seed: int,
    ) -> Image.Image:
        """Regenerate the exact augmented image from the label + augmentation parameters."""
        label_518 = letterbox_pil(label_image, self.canonical_size)
        if aug_name == "catalog":
            return label_518
        aug_fn = _AUG_REGISTRY.get(aug_name)
        if aug_fn is None:
            return label_518
        rng = random.Random(aug_seed)
        return aug_fn(label_518, rng)

    def compare(
        self,
        query: Image.Image,
        label_image: Image.Image,
        aug_name: str,
        aug_seed: int,
        dino_similarity: float,
    ) -> SiftResultV3:
        """Compare query against the regenerated augmented crop."""
        candidate = self.regenerate_augment(label_image, aug_name, aug_seed)

        compare_size = 384
        query_gray = cv2.cvtColor(
            np.array(query.resize((compare_size, compare_size))), cv2.COLOR_RGB2GRAY
        )
        candidate_gray = cv2.cvtColor(
            np.array(candidate.resize((compare_size, compare_size))), cv2.COLOR_RGB2GRAY
        )

        sift = cv2.SIFT_create(nfeatures=self.max_features)
        qkp, qdesc = sift.detectAndCompute(query_gray, None)
        ckp, cdesc = sift.detectAndCompute(candidate_gray, None)

        if qdesc is None or cdesc is None or len(qkp) < 4 or len(ckp) < 4:
            return SiftResultV3(score=dino_similarity * 0.4, inliers=0, inlier_ratio=0.0)

        matches = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5), dict(checks=50)
        ).knnMatch(qdesc, cdesc, k=2)
        good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]

        inliers = 0
        if len(good) >= 4:
            src = np.float32([qkp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([ckp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers = int(mask.sum())

        ratio = inliers / max(1, len(good))
        mass = min(1.0, inliers / 300.0)
        score = dino_similarity * 0.4 + mass * 0.3 + ratio * 0.3
        return SiftResultV3(score=float(score), inliers=inliers, inlier_ratio=float(ratio))
