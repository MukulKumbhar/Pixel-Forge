"""
ImageLab Pro - Quantitative Image Quality & Radiometric Metrics
Calculates PSNR, MSE, SSIM, Shannon Entropy, Dynamic Range, and Multi-Channel Histograms.
"""

import cv2
import numpy as np


def calculate_mse(img_orig: np.ndarray, img_proc: np.ndarray) -> float:
    """
    Mean Squared Error: MSE = (1 / MN) * sum (I_orig - I_proc)^2
    """
    # Ensure identical shape and channels for comparison
    o, p = _align_image_shapes(img_orig, img_proc)
    diff = o.astype(np.float64) - p.astype(np.float64)
    return float(np.mean(diff ** 2))


def calculate_psnr(img_orig: np.ndarray, img_proc: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR in decibels dB):
    PSNR = 10 * log10( (MAX_I)^2 / MSE )
    """
    mse = calculate_mse(img_orig, img_proc)
    if mse < 1e-10:
        return 99.9  # Infinitely high quality (lossless match)
    max_val = 255.0
    return float(10.0 * np.log10((max_val ** 2) / mse))


def calculate_ssim(img_orig: np.ndarray, img_proc: np.ndarray) -> float:
    """
    Structural Similarity Index Metric (SSIM):
    SSIM = [ (2*mu_x*mu_y + c1)(2*sigma_xy + c2) ] / [ (mu_x^2 + mu_y^2 + c1)(sigma_x^2 + sigma_y^2 + c2) ]
    Measures luminance, contrast, and structural preservation.
    """
    o, p = _align_image_shapes(img_orig, img_proc)
    if o.ndim == 3:
        o = cv2.cvtColor(o, cv2.COLOR_RGB2GRAY)
        p = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY)

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
    """
    Shannon Image Information Entropy:
    H = - sum_{k} p(k) * log2(p(k))
    Quantifies average information content and contrast richness (in bits/pixel).
    """
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
    """
    Computes normalized histogram distributions for rendering comparative charts.
    Returns dictionary with channel keys: 'r', 'g', 'b', or 'luminance'.
    """
    bins = 256
    if image.ndim == 3 and image.shape[2] >= 3:
        hist_r, _ = np.histogram(image[:, :, 0], bins=bins, range=[0, 256])
        hist_g, _ = np.histogram(image[:, :, 1], bins=bins, range=[0, 256])
        hist_b, _ = np.histogram(image[:, :, 2], bins=bins, range=[0, 256])
        # Luminance
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
    """
    Helper to align spatial dimensions and channels if a pipeline step altered them.
    """
    # Normalize RGBA to RGB for metric comparisons
    if img_a.ndim == 3 and img_a.shape[2] == 4:
        img_a = cv2.cvtColor(img_a, cv2.COLOR_RGBA2RGB)
    if img_b.ndim == 3 and img_b.shape[2] == 4:
        img_b = cv2.cvtColor(img_b, cv2.COLOR_RGBA2RGB)

    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]

    # Resize img_b to match img_a if dimensions changed (e.g. downsampling or cropping)
    if (ha, wa) != (hb, wb):
        img_b = cv2.resize(img_b, (wa, ha), interpolation=cv2.INTER_LINEAR)

    # Match channel counts
    if img_a.ndim == 3 and img_b.ndim == 2:
        img_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2RGB)
    elif img_a.ndim == 2 and img_b.ndim == 3:
        img_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)

    return img_a, img_b
