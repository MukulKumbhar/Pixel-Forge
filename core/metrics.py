"""
Image quality and radiometric metric calculations (PSNR, MSE, SSIM, Entropy).
"""

import cv2
import numpy as np


def calculate_mse(img_orig: np.ndarray, img_proc: np.ndarray) -> float:
    """Mean Squared Error between original and processed images."""
    o, p = _align_image_shapes(img_orig, img_proc)
    diff = o.astype(np.float64) - p.astype(np.float64)
    return float(np.mean(diff ** 2))


def calculate_psnr(img_orig: np.ndarray, img_proc: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in decibels (dB)."""
    mse = calculate_mse(img_orig, img_proc)
    if mse < 1e-10:
        return 99.9
    max_val = 255.0
    return float(10.0 * np.log10((max_val ** 2) / mse))


def calculate_ssim(img_orig: np.ndarray, img_proc: np.ndarray) -> float:
    """Structural Similarity Index Metric (SSIM)."""
    o, p = _align_image_shapes(img_orig, img_proc)
    if o.ndim == 3:
        o = cv2.cvtColor(o, cv2.COLOR_RGB2GRAY)
        p = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY)

    h_max = max(o.shape[:2])
    if h_max > 2048:
        scale_ssim = 2048.0 / h_max
        new_w = max(16, int(o.shape[1] * scale_ssim))
        new_h = max(16, int(o.shape[0] * scale_ssim))
        o = cv2.resize(o, (new_w, new_h), interpolation=cv2.INTER_AREA)
        p = cv2.resize(p, (new_w, new_h), interpolation=cv2.INTER_AREA)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    o = o.astype(np.float64)
    p = p.astype(np.float64)

    mu_x = cv2.GaussianBlur(o, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(p, (11, 11), 1.5)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = cv2.GaussianBlur(o ** 2, (11, 11), 1.5) - mu_x_sq
    sigma_y_sq = cv2.GaussianBlur(p ** 2, (11, 11), 1.5) - mu_y_sq
    sigma_xy = cv2.GaussianBlur(o * p, (11, 11), 1.5) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / (denominator + 1e-10)
    return float(np.mean(ssim_map))


def calculate_entropy(image: np.ndarray) -> float:
    """Shannon image information entropy in bits per pixel."""
    if image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    elif image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.squeeze()

    hist, _ = np.histogram(gray, bins=256, range=[0, 256])
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist[hist > 0] / total
    return float(-np.sum(p * np.log2(p)))


def compute_histograms(image: np.ndarray) -> dict:
    """Compute normalized histogram distributions for channels and luminance."""
    bins = 256
    if image.ndim == 3 and image.shape[2] >= 3:
        hist_r, _ = np.histogram(image[:, :, 0], bins=bins, range=[0, 256])
        hist_g, _ = np.histogram(image[:, :, 1], bins=bins, range=[0, 256])
        hist_b, _ = np.histogram(image[:, :, 2], bins=bins, range=[0, 256])
        if image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hist_l, _ = np.histogram(gray, bins=bins, range=[0, 256])
        return {
            "mode": "rgb",
            "r": hist_r,
            "g": hist_g,
            "b": hist_b,
            "luminance": hist_l
        }
    else:
        gray = image.squeeze()
        hist_l, _ = np.histogram(gray, bins=bins, range=[0, 256])
        return {
            "mode": "grayscale",
            "luminance": hist_l
        }


def _align_image_shapes(img_a: np.ndarray, img_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align dimensions and channel formats for comparison."""
    if img_a.ndim == 3 and img_a.shape[2] == 4:
        img_a = cv2.cvtColor(img_a, cv2.COLOR_RGBA2RGB)
    if img_b.ndim == 3 and img_b.shape[2] == 4:
        img_b = cv2.cvtColor(img_b, cv2.COLOR_RGBA2RGB)

    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]

    if (ha, wa) != (hb, wb):
        img_b = cv2.resize(img_b, (wa, ha), interpolation=cv2.INTER_LINEAR)

    if img_a.ndim == 3 and img_b.ndim == 2:
        img_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2RGB)
    elif img_a.ndim == 2 and img_b.ndim == 3:
        img_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)

    return img_a, img_b
