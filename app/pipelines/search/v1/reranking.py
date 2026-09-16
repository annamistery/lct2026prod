from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SiftResult:
    score: float
    inliers: int
    inlier_ratio: float


class SiftReranker:
    def __init__(self, max_features: int = 1000):
        self.max_features = max_features

    def compare(self, query: Image.Image, candidate: Image.Image, dino_similarity: float) -> SiftResult:
        query_gray = cv2.cvtColor(np.array(query.resize((384, 384))), cv2.COLOR_RGB2GRAY)
        candidate_gray = cv2.cvtColor(np.array(candidate.resize((384, 384))), cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create(nfeatures=self.max_features)
        query_points, query_descriptors = sift.detectAndCompute(query_gray, None)
        candidate_points, candidate_descriptors = sift.detectAndCompute(candidate_gray, None)
        if query_descriptors is None or candidate_descriptors is None or len(query_points) < 4 or len(candidate_points) < 4:
            return SiftResult(score=dino_similarity * 0.4, inliers=0, inlier_ratio=0.0)
        matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50)).knnMatch(query_descriptors, candidate_descriptors, k=2)
        good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
        inliers = 0
        if len(good) >= 4:
            source = np.float32([query_points[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
            target = np.float32([candidate_points[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers = int(mask.sum())
        ratio = inliers / max(1, len(good))
        mass = min(1.0, inliers / 300.0)
        score = dino_similarity * 0.4 + mass * 0.3 + ratio * 0.3
        return SiftResult(score=float(score), inliers=inliers, inlier_ratio=float(ratio))
