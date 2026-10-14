"""Radiometric, tone mapping, and contrast calibration algorithms."""

import cv2
import numpy as np


def image_negative(image: np.ndarray) -> np.ndarray:
    """Computes photographic negative (s = 255 - r)."""
    return 255 - image


def log_transform(image: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Applies logarithmic dynamic range compression."""
    c_norm = (255.0 / np.log(1.0 + 255.0)) * float(c)
    s = c_norm * np.log(1.0 + image.astype(np.float32))
    return np.clip(s, 0, 255).astype(np.uint8)


def power_law_gamma(image: np.ndarray, gamma: float = 1.0, c: float = 1.0) -> np.ndarray:
    """Power-law gamma correction via precomputed 256-entry lookup table."""
    gamma = max(0.01, float(gamma))
    c = max(0.01, float(c))
    r = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip(c * (r ** gamma) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(image, lut)


def contrast_stretch_piecewise(
    image: np.ndarray,
    r1: int = 70,
    s1: int = 20,
    r2: int = 180,
    s2: int = 230
) -> np.ndarray:
    """Piecewise linear contrast stretching across shadows, midtones, and highlights."""
    r1, r2 = min(r1, r2 - 1), max(r2, r1 + 1)
    s1, s2 = np.clip([s1, s2], 0, 255)
    lut = np.zeros(256, dtype=np.uint8)

    # Shadow segment
    slope1 = s1 / max(1, r1)
    lut[:r1] = np.clip(slope1 * np.arange(r1), 0, 255)

    # Midtone segment
    slope2 = (s2 - s1) / max(1, r2 - r1)
    lut[r1:r2] = np.clip(s1 + slope2 * (np.arange(r1, r2) - r1), 0, 255)

    # Highlight segment
    slope3 = (255 - s2) / max(1, 255 - r2)
    lut[r2:] = np.clip(s2 + slope3 * (np.arange(r2, 256) - r2), 0, 255)

    return cv2.LUT(image, lut)


def bit_plane_slice(image: np.ndarray, plane: int = 7, mode: str = "single") -> np.ndarray:
    """Extracts or reconstructs individual binary bit planes (0=LSB, 7=MSB)."""
    plane = int(np.clip(plane, 0, 7))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()

    if mode == "single":
        return (((gray >> plane) & 1) * 255).astype(np.uint8)

    mask = sum(1 << b for b in range(plane, 8))
    return (gray & mask).astype(np.uint8)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Spreads intensity values evenly using cumulative distribution function."""
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        return cv2.equalizeHist(image.squeeze())

    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def histogram_specification(image: np.ndarray, target_profile: str = "gaussian") -> np.ndarray:
    """Transforms image histogram to match a target analytical distribution."""
    def match_channel(src: np.ndarray, target_cdf: np.ndarray) -> np.ndarray:
        src_hist, _ = np.histogram(src.flatten(), 256, [0, 256])
        src_cdf = src_hist.cumsum().astype(np.float64)
        src_cdf /= src_cdf[-1]

        lut = np.zeros(256, dtype=np.uint8)
        tgt_idx = 0
        for src_val in range(256):
            while tgt_idx < 255 and target_cdf[tgt_idx] < src_cdf[src_val]:
                tgt_idx += 1
            lut[src_val] = tgt_idx
        return cv2.LUT(src, lut)

    x = np.linspace(0, 255, 256)
    if target_profile == "gaussian":
        pdf = np.exp(-((x - 128) ** 2) / (2 * (40 ** 2)))
    elif target_profile == "bimodal":
        pdf = np.exp(-((x - 64) ** 2) / (2 * (25 ** 2))) + np.exp(-((x - 192) ** 2) / (2 * (25 ** 2)))
    else:
        pdf = np.ones(256)

    target_cdf = pdf.cumsum()
    target_cdf /= target_cdf[-1]

    if image.ndim == 2:
        return match_channel(image, target_cdf)

    matched = np.zeros_like(image)
    for c in range(3):
        matched[:, :, c] = match_channel(image[:, :, c], target_cdf)
    return matched


def brightness_contrast(image: np.ndarray, brightness: int = 0, contrast: int = 0) -> np.ndarray:
    """Adjusts linear brightness offset and contrast gain."""
    alpha = (contrast + 100.0) / 100.0
    alpha = (alpha ** 1.4) if contrast >= 0 else alpha
    return cv2.convertScaleAbs(image, alpha=alpha, beta=float(brightness))


def solarize_effect(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Inverts pixels exceeding threshold (Sabattier darkroom effect)."""
    img_c = image.copy()
    mask = img_c > threshold
    img_c[mask] = 255 - img_c[mask]
    return img_c


def posterize_levels(image: np.ndarray, levels: int = 4) -> np.ndarray:
    """Quantizes continuous color gradations into discrete graphic bands."""
    levels = max(2, min(32, int(levels)))
    step = 255.0 / (levels - 1)
    quantized = np.round(image / (256.0 / levels)) * step
    return np.clip(quantized, 0, 255).astype(np.uint8)
