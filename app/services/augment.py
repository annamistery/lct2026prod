"""Camera-like augmentations for domain-gap robustness (LCT2026).

Exact 23 augmentation functions ported from dataset_v5/augment.py.
Generates an augmented cloud of crops (1 original + 23 augs * 5 variants = 116 images).
"""

from __future__ import annotations

import io
import random
from collections.abc import Callable

import cv2
import numpy as np
from PIL import Image, ImageEnhance

AugFn = Callable[[Image.Image, random.Random | None], Image.Image]

_AUG_REGISTRY: dict[str, AugFn] = {}


def register(name: str):
    def decorator(fn: AugFn) -> AugFn:
        _AUG_REGISTRY[name] = fn
        return fn

    return decorator


def _to_array(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _rng(rng: random.Random | None) -> random.Random:
    return rng if rng is not None else random.Random()


@register("orig")
def orig(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    return image.copy()


@register("perspective")
def perspective(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    r = _rng(rng)
    img = _to_array(image)
    h, w = img.shape[:2]
    max_shift = 0.12
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst = np.float32(
        [
            [r.uniform(-max_shift, max_shift) * w, r.uniform(-max_shift, max_shift) * h],
            [w + r.uniform(-max_shift, max_shift) * w, r.uniform(-max_shift, max_shift) * h],
            [r.uniform(-max_shift, max_shift) * w, h + r.uniform(-max_shift, max_shift) * h],
            [w + r.uniform(-max_shift, max_shift) * w, h + r.uniform(-max_shift, max_shift) * h],
        ]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return _to_pil(out)


@register("rotate")
def rotate(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    r = _rng(rng)
    angle = r.uniform(-15, 15)
    return image.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=None)


@register("brightness_up")
def brightness_up(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    r = _rng(rng)
    factor = r.uniform(1.25, 1.8)
    return ImageEnhance.Brightness(image).enhance(factor)


@register("brightness_down")
def brightness_down(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    r = _rng(rng)
    factor = r.uniform(0.4, 0.75)
    return ImageEnhance.Brightness(image).enhance(factor)


@register("contrast_up")
def contrast_up(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    r = _rng(rng)
    factor = r.uniform(1.3, 2.0)
    return ImageEnhance.Contrast(image).enhance(factor)


@register("contrast_down")
def contrast_down(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    r = _rng(rng)
    factor = r.uniform(0.4, 0.75)
    return ImageEnhance.Contrast(image).enhance(factor)


@register("jpeg_low")
def jpeg_low(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    r = _rng(rng)
    quality = int(r.uniform(15, 45))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


@register("gaussian_blur")
def gaussian_blur(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    r = _rng(rng)
    img = _to_array(image)
    k = r.choice([3, 5, 7])
    out = cv2.GaussianBlur(img, (k, k), 0)
    return _to_pil(out)


@register("motion_blur")
def motion_blur(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    r = _rng(rng)
    size = r.choice([5, 7, 9])
    angle = r.choice([0, 45, 90, 135])
    kernel = np.zeros((size, size))
    if angle in (0, 180):
        kernel[size // 2, :] = np.ones(size)
    elif angle in (90, 270):
        kernel[:, size // 2] = np.ones(size)
    else:
        for i in range(size):
            kernel[i, i] = 1
    kernel = kernel / kernel.sum()
    img = cv2.filter2D(_to_array(image), -1, kernel)
    return _to_pil(img)


@register("noise")
def noise(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    r = _rng(rng)
    img = _to_array(image).astype(np.float32)
    sigma = r.uniform(10, 35)
    np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1) if rng else None)
    img += np_rng.normal(0, sigma, img.shape)
    return _to_pil(img)


@register("color_jitter")
def color_jitter(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    r = _rng(rng)
    img = _to_array(image).astype(np.float32)
    gains = [r.uniform(0.75, 1.25) for _ in range(3)]
    for i, g in enumerate(gains):
        img[..., i] *= g
    return _to_pil(img)


@register("wb_warm")
def wb_warm(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    r = _rng(rng)
    img = _to_array(image).astype(np.float32)
    img[..., 0] *= r.uniform(1.1, 1.4)
    img[..., 2] *= r.uniform(0.75, 0.95)
    return _to_pil(img)


@register("wb_cool")
def wb_cool(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    r = _rng(rng)
    img = _to_array(image).astype(np.float32)
    img[..., 0] *= r.uniform(0.75, 0.95)
    img[..., 2] *= r.uniform(1.1, 1.4)
    return _to_pil(img)


@register("glare")
def glare(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    r = _rng(rng)
    img = _to_array(image)
    h, w = img.shape[:2]
    overlay = np.zeros_like(img, dtype=np.float32)
    cx = int(r.uniform(0.1, 0.9) * w)
    cy = int(r.uniform(0.1, 0.9) * h)
    ax = int(r.uniform(0.1, 0.35) * w)
    ay = int(r.uniform(0.1, 0.35) * h)
    angle = r.uniform(0, 360)
    intensity = r.uniform(140, 255)
    cv2.ellipse(overlay, (cx, cy), (ax, ay), angle, 0, 360, (intensity, intensity, intensity), -1)
    k = max(5, min(ax, ay) | 1)
    overlay = cv2.GaussianBlur(overlay, (k, k), 0)
    alpha = r.uniform(0.25, 0.55)
    blended = img.astype(np.float32) * (1 - alpha) + overlay * alpha
    return _to_pil(blended)


@register("local_overexposure")
def local_overexposure(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    r = _rng(rng)
    img = _to_array(image)
    h, w = img.shape[:2]
    x1 = int(r.uniform(0, 0.6) * w)
    y1 = int(r.uniform(0, 0.6) * h)
    x2 = min(w, x1 + int(r.uniform(0.25, 0.6) * w))
    y2 = min(h, y1 + int(r.uniform(0.25, 0.6) * h))
    region = img[y1:y2, x1:x2].astype(np.float32)
    boost = r.uniform(50, 130)
    region = np.clip(region + boost, 0, 255)
    img = img.copy()
    img[y1:y2, x1:x2] = region.astype(np.uint8)
    return _to_pil(img)


@register("gamma_low")
def gamma_low(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    r = _rng(rng)
    gamma = r.uniform(0.5, 0.8)
    img = _to_array(image)
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    out = cv2.LUT(img, table)
    return _to_pil(out)


@register("gamma_high")
def gamma_high(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    r = _rng(rng)
    gamma = r.uniform(1.8, 2.5)
    img = _to_array(image)
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    out = cv2.LUT(img, table)
    return _to_pil(out)


@register("saturation_down")
def saturation_down(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    r = _rng(rng)
    factor = r.uniform(0.3, 0.7)
    return ImageEnhance.Color(image).enhance(factor)


@register("saturation_up")
def saturation_up(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    r = _rng(rng)
    factor = r.uniform(1.4, 2.0)
    return ImageEnhance.Color(image).enhance(factor)


@register("combined_1")
def combined_1(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    r = _rng(rng)
    image = perspective(image, r)
    image = brightness_down(image, r)
    image = gaussian_blur(image, r)
    return image


@register("combined_2")
def combined_2(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    r = _rng(rng)
    image = rotate(image, r)
    image = contrast_down(image, r)
    image = jpeg_low(image, r)
    return image


@register("combined_4")
def combined_4(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    r = _rng(rng)
    image = glare(image, r)
    image = brightness_up(image, r)
    image = noise(image, r)
    return image


@register("combined_5")
def combined_5(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    r = _rng(rng)
    image = perspective(image, r)
    image = local_overexposure(image, r)
    image = saturation_down(image, r)
    image = jpeg_low(image, r)
    return image


@register("cylinder_warp")
def cylinder_warp(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    """Simulate the cylindrical curvature of a wine bottle."""
    r = _rng(rng)
    img = _to_array(image)
    h, w = img.shape[:2]

    # Arc angle for bottle curvature: 0.45 to 0.85 rad (approx 25 to 50 degrees)
    alpha = r.uniform(0.45, 0.85)
    sin_alpha = np.sin(alpha)

    # Normalized x coordinates across width [-1, 1]
    x_norm = np.linspace(-1.0, 1.0, w, dtype=np.float32)

    # Inverse cylindrical mapping to sample from the flat image
    x_src = np.arcsin(np.clip(x_norm * sin_alpha, -0.999, 0.999)) / alpha
    map_x = ((x_src + 1.0) * 0.5 * (w - 1)).astype(np.float32)

    # Create 2D coordinate maps
    map_x_grid = np.tile(map_x, (h, 1))
    map_y_grid = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

    out = cv2.remap(img, map_x_grid, map_y_grid, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return _to_pil(out)


@register("shadow")
def shadow(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    """Simulate directional bottle shadow across the label."""
    r = _rng(rng)
    img = _to_array(image).astype(np.float32)
    h, w = img.shape[:2]

    direction = r.choice(["left", "right"])
    min_factor = r.uniform(0.35, 0.65)
    gradient = np.linspace(1.0, min_factor, w, dtype=np.float32)
    if direction == "right":
        gradient = gradient[::-1]

    img *= gradient[None, :, None]
    return _to_pil(img)


# --- v3 geometric augmentations (zero-fill borders) ---

@register("view_from_above")
def view_from_above(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    """Trapezoidal distortion: top wider, bottom narrower (camera above label)."""
    r = _rng(rng)
    img = _to_array(image)
    h, w = img.shape[:2]
    top_shift = r.uniform(0.02, 0.06)
    bot_shift = r.uniform(0.08, 0.15)
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst = np.float32([
        [-top_shift * w, 0],
        [w + top_shift * w, 0],
        [bot_shift * w, h],
        [w - bot_shift * w, h],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return _to_pil(out)


@register("view_from_below")
def view_from_below(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    """Trapezoidal distortion: bottom wider, top narrower (camera below label)."""
    r = _rng(rng)
    img = _to_array(image)
    h, w = img.shape[:2]
    top_shift = r.uniform(0.08, 0.15)
    bot_shift = r.uniform(0.02, 0.06)
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst = np.float32([
        [top_shift * w, 0],
        [w - top_shift * w, 0],
        [-bot_shift * w, h],
        [w + bot_shift * w, h],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return _to_pil(out)


@register("tilt_small")
def tilt_small(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    """Small rotation up to 5 degrees with zero-fill corners."""
    r = _rng(rng)
    angle = r.uniform(-5, 5)
    return image.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))


@register("combined_geo_light")
def combined_geo_light(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    """Combined: cylinder_warp + shadow + brightness_down."""
    r = _rng(rng)
    image = cylinder_warp(image, r)
    image = shadow(image, r)
    image = brightness_down(image, r)
    return image


# Base 23 augmentations used for catalog ingestion (exact 116 images cloud)
BASE_CATALOG_AUG_NAMES: list[str] = [
    "perspective", "rotate", "brightness_up", "brightness_down", "contrast_up", "contrast_down",
    "jpeg_low", "gaussian_blur", "motion_blur", "noise", "color_jitter", "wb_warm", "wb_cool",
    "glare", "local_overexposure", "gamma_low", "gamma_high", "saturation_down", "saturation_up",
    "combined_1", "combined_2", "combined_4", "combined_5",
]

# v3 augmentations: emphasis on geometric distortions + zero-fill
BASE_CATALOG_AUG_NAMES_V3: list[str] = [
    "perspective", "rotate", "cylinder_warp", "view_from_above", "view_from_below",
    "tilt_small", "shadow",
    "brightness_up", "brightness_down", "contrast_up", "contrast_down",
    "jpeg_low", "gaussian_blur", "motion_blur", "noise",
    "color_jitter", "wb_warm", "wb_cool",
    "glare", "local_overexposure",
    "gamma_low", "gamma_high",
    "combined_geo_light",
]

AUGMENTATION_NAMES: list[str] = BASE_CATALOG_AUG_NAMES


def generate_augmented_cloud(canonical_label: Image.Image, variants_per_aug: int = 5, seed: int = 42) -> list[tuple[str, Image.Image]]:
    """Generate the full cloud of augmentations from a 256x256 canonical label.

    Returns:
        List of tuples: (aug_name, image).
        The first element is always ('catalog', original_copy).
        Followed by 23 aug types * variants_per_aug = 115 augmented images.
        Total = 116 images.
    """
    results: list[tuple[str, Image.Image]] = [("catalog", canonical_label.copy())]
    rng = random.Random(seed)

    for aug_name in AUGMENTATION_NAMES:
        aug_fn = _AUG_REGISTRY[aug_name]
        for _ in range(variants_per_aug):
            aug_img = aug_fn(canonical_label, rng)
            results.append((aug_name, aug_img))

    return results


def generate_augmented_cloud_v3(
    canonical_label: Image.Image, variants_per_aug: int = 5, seed: int = 42,
) -> list[tuple[str, int, Image.Image]]:
    """Generate v3 augmented cloud from a 518x518 letterbox canonical label.

    Returns:
        List of tuples: (aug_name, variant_seed, image).
        The first element is always ('catalog', seed, original_copy).
        Followed by 23 aug types * variants_per_aug = 115 augmented images.
        Total = 116 images.

    The variant_seed can be used to reproduce the augmentation deterministically.
    """
    results: list[tuple[str, int, Image.Image]] = [("catalog", seed, canonical_label.copy())]
    rng = random.Random(seed)

    for aug_name in BASE_CATALOG_AUG_NAMES_V3:
        aug_fn = _AUG_REGISTRY[aug_name]
        for _variant_i in range(variants_per_aug):
            variant_seed = rng.randint(0, 2**31 - 1)
            variant_rng = random.Random(variant_seed)
            aug_img = aug_fn(canonical_label, variant_rng)
            results.append((aug_name, variant_seed, aug_img))

    return results


AUGMENTATION_LABELS_RU: dict[str, str] = {
    "perspective": "3D-перспектива (наклон камеры)",
    "rotate": "Поворот ракурса",
    "cylinder_warp": "Цилиндрический изгиб бутылки",
    "view_from_above": "Вид сверху (трапеция)",
    "view_from_below": "Вид снизу (трапеция)",
    "tilt_small": "Малый наклон (до 5°)",
    "combined_geo_light": "Цилиндр + тень + затемнение",
    "glare": "Световой блик",
    "shadow": "Падающая тень",
    "local_overexposure": "Локальный засвет",
    "brightness_down": "Приглушённый свет",
    "brightness_up": "Яркое освещение",
    "contrast_down": "Сниженный контраст",
    "contrast_up": "Повышенный контраст",
    "motion_blur": "Смаз движения руки",
    "gaussian_blur": "Нечёткий фокус",
    "noise": "Шум матрицы",
    "jpeg_low": "Артефакты сжатия камеры",
    "color_jitter": "Цветовое смещение",
    "wb_warm": "Тёплый баланс белого",
    "wb_cool": "Холодный баланс белого",
    "gamma_low": "Затемнение гаммы",
    "gamma_high": "Осветление гаммы",
    "saturation_down": "Приглушённая цветность",
    "saturation_up": "Насыщенные цвета",
}


def apply_random_hard_augmentation(image: Image.Image, rng: random.Random | None = None) -> tuple[Image.Image, list[str]]:
    """Apply a realistic random combination of hard distortions to simulate real-world mobile camera capture."""
    r = _rng(rng)
    out = image.copy()
    applied: list[str] = []

    # 1. Geometric distortion (cylinder bottle curve, perspective tilt, or rotation)
    geo_pool = [k for k in ["cylinder_warp", "perspective", "rotate"] if k in _AUG_REGISTRY]
    if geo_pool:
        geo_choice = r.choice(geo_pool)
        out = _AUG_REGISTRY[geo_choice](out, r)
        applied.append(AUGMENTATION_LABELS_RU.get(geo_choice, geo_choice))

    # 2. Lighting / Exposure distortion (shadow, glare, overexposure, dim light)
    light_pool = [k for k in ["shadow", "glare", "local_overexposure", "brightness_down", "contrast_down"] if k in _AUG_REGISTRY]
    if light_pool:
        light_choice = r.choice(light_pool)
        out = _AUG_REGISTRY[light_choice](out, r)
        applied.append(AUGMENTATION_LABELS_RU.get(light_choice, light_choice))

    # 3. Camera sensor optics & noise (motion blur, defocus, sensor noise, compression)
    sensor_pool = [k for k in ["motion_blur", "gaussian_blur", "noise", "jpeg_low", "color_jitter"] if k in _AUG_REGISTRY]
    if sensor_pool:
        sensor_count = min(len(sensor_pool), r.choice([1, 2]))
        sensor_choices = r.sample(sensor_pool, k=sensor_count)
        for sc in sensor_choices:
            out = _AUG_REGISTRY[sc](out, r)
            applied.append(AUGMENTATION_LABELS_RU.get(sc, sc))

    return out, applied
