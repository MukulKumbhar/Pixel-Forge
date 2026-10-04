"""
=============================================================================
ImageLab Pro - Stochastic Denoising & Image Restoration
Aligned with Academic DIP Lab Experiments (Exp 07)
=============================================================================
Covers:
- Degradation Models: Gaussian, Salt & Pepper, Speckle Noise
- Linear Spatial Filtering: 3x3 Averaging (Mean) Filter (Exp 07 Part 1)
- Non-Linear Spatial Filtering: Median Filter (Exp 07 Part 2)
- Order-Statistic Filters: Min, Max, Midpoint
- Geometric, Harmonic, Contraharmonic (Order Q), and Alpha-Trimmed Filters
=============================================================================
"""

import cv2
import numpy as np


# ============================================================
# NOISE DEGRADATION MODEL: Gaussian Noise
# Aim: Simulate thermal sensor noise in camera sensors
# Formula: g(x, y) = f(x, y) + n(x, y), where n ~ N(mean, sigma^2)
# ============================================================
def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, sigma: float = 25.0) -> np.ndarray:
    """
    Adds zero-mean Gaussian electronic noise to the image.
    """
    noise = np.random.normal(mean, sigma, image.shape)
    degraded = image.astype(np.float32) + noise
    return np.clip(degraded, 0, 255).astype(np.uint8)


# ============================================================
# NOISE DEGRADATION MODEL: Bipolar Salt & Pepper Impulse Noise
# Aim: Simulate transmission bit errors or dead sensor elements
# Formula: Salt = 255 (prob = salt_prob), Pepper = 0 (prob = pepper_prob)
# ============================================================
def add_salt_and_pepper_noise(image: np.ndarray, salt_prob: float = 0.02, pepper_prob: float = 0.02) -> np.ndarray:
    """
    Randomly drops pixel intensities to pure white (salt) or black (pepper).
    """
    noisy = image.copy()
    h, w = image.shape[:2]
    total_pixels = h * w

    # Step 1: Inject white salt pixels (255)
    num_salt = int(total_pixels * salt_prob)
    if num_salt > 0:
        ys = np.random.randint(0, h, num_salt)
        xs = np.random.randint(0, w, num_salt)
        noisy[ys, xs] = 255

    # Step 2: Inject black pepper pixels (0)
    num_pepper = int(total_pixels * pepper_prob)
    if num_pepper > 0:
        yp = np.random.randint(0, h, num_pepper)
        xp = np.random.randint(0, w, num_pepper)
        noisy[yp, xp] = 0

    return noisy


# ============================================================
# NOISE DEGRADATION MODEL: Multiplicative Speckle Noise
# Aim: Simulate coherent wave interference in ultrasound and radar
# Formula: g(x, y) = f(x, y) + f(x, y) * n
# ============================================================
def add_speckle_noise(image: np.ndarray, variance: float = 0.04) -> np.ndarray:
    """
    Adds multiplicative speckle noise characteristic of coherent ultrasound imaging.
    """
    std = np.sqrt(variance)
    noise = np.random.normal(0, std, image.shape)
    degraded = image.astype(np.float32) + (image.astype(np.float32) * noise)
    return np.clip(degraded, 0, 255).astype(np.uint8)


# ============================================================
# DIP EXPERIMENT REF: EXP 07 - PART 1 (LINEAR FILTERING)
# Algorithm: Averaging (Mean) Filter
# Aim: Replace each pixel with the average of its neighborhood
# Formula: f_hat(x, y) = (1 / (m * n)) * sum(neighborhood)
# ============================================================
def arithmetic_mean_filter(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Arithmetic Mean (Box) Filter:
    Smooths local pixel variations by taking the unweighted average
    of all pixels within a k x k sliding kernel window.
    """
    # Kernel size must be an odd positive integer (3, 5, 7, ...)
    ksize = max(3, ksize if ksize % 2 != 0 else ksize + 1)
    return cv2.blur(image, (ksize, ksize))


# ============================================================
# DIP EXPERIMENT REF: EXP 07 - PART 2 (NON-LINEAR FILTERING)
# Algorithm: 3x3 Median Filter
# Aim: Replace each pixel with the statistical median of its neighborhood
# Formula: f_hat(x, y) = median{ g(s, t) in S_xy }
# ============================================================
def median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Median Filter:
    The standard non-linear filter for eliminating salt-and-pepper noise
    without blurring high-frequency edge details.
    """
    ksize = max(3, ksize if ksize % 2 != 0 else ksize + 1)
    return cv2.medianBlur(image, ksize)


# ============================================================
# ALGORITHM: Geometric Mean Filter
# Aim: Filter Gaussian noise with better edge sharpness retention
# Formula: f_hat(x, y) = [ prod g(s, t) ]^(1 / mn)
# ============================================================
def geometric_mean_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Geometric Mean Filter:
    Uses logarithmic averaging: exp( (1/mn) * sum( log(g + eps) ) )
    """
    ksize = max(3, ksize if ksize % 2 != 0 else ksize + 1)
    eps = 1e-4

    def _geo_2d(channel):
        log_ch = np.log(channel.astype(np.float32) + eps)
        kernel = np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize)
        filtered_log = cv2.filter2D(log_ch, -1, kernel, borderType=cv2.BORDER_REFLECT)
        return np.clip(np.exp(filtered_log) - eps, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return _geo_2d(image)

    out = np.zeros_like(image)
    for c in range(image.shape[2]):
        out[:, :, c] = _geo_2d(image[:, :, c])
    return out


# ============================================================
# ALGORITHM: Contraharmonic Mean Filter (Order Q)
# Aim: Selectively eliminate salt noise (Q > 0) or pepper noise (Q < 0)
# Formula: f_hat = sum( g^(Q+1) ) / sum( g^Q )
# ============================================================
def contraharmonic_mean_filter(image: np.ndarray, ksize: int = 3, Q: float = 1.5) -> np.ndarray:
    """
    Contraharmonic Filter:
    - Q > 0: Eliminates pepper noise (dark spots)
    - Q < 0: Eliminates salt noise (white spots)
    - Q = 0: Simplifies to arithmetic mean filter
    """
    ksize = max(3, ksize if ksize % 2 != 0 else ksize + 1)
    eps = 1e-4

    def _chm_2d(channel):
        ch_f = channel.astype(np.float64) + eps
        kernel = np.ones((ksize, ksize), dtype=np.float64)

        num = cv2.filter2D(np.power(ch_f, Q + 1.0), -1, kernel, borderType=cv2.BORDER_REFLECT)
        den = cv2.filter2D(np.power(ch_f, Q), -1, kernel, borderType=cv2.BORDER_REFLECT)

        result = np.where(den > eps, num / den, ch_f)
        return np.clip(result, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return _chm_2d(image)

    out = np.zeros_like(image)
    for c in range(image.shape[2]):
        out[:, :, c] = _chm_2d(image[:, :, c])
    return out


# ============================================================
# ALGORITHM: Order-Statistic Filter (Min / Max / Midpoint)
# Aim: Non-linear morphological extrema filtering
# ============================================================
def order_statistic_filter(image: np.ndarray, ksize: int = 3, mode: str = "median") -> np.ndarray:
    """
    Order-Statistic Filters:
    - 'min': Eradicates white salt noise by choosing darkest neighbor
    - 'max': Eradicates dark pepper noise by choosing brightest neighbor
    - 'midpoint': Computes average of min and max pixel: (min + max) / 2
    """
    ksize = max(3, ksize if ksize % 2 != 0 else ksize + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))

    if mode == "min":
        return cv2.erode(image, kernel)
    elif mode == "max":
        return cv2.dilate(image, kernel)
    elif mode == "midpoint":
        min_img = cv2.erode(image, kernel).astype(np.float32)
        max_img = cv2.dilate(image, kernel).astype(np.float32)
        return np.clip((min_img + max_img) / 2.0, 0, 255).astype(np.uint8)
    else:
        return cv2.medianBlur(image, ksize)


# Backward-compatible alias
order_statistic_min_max_filter = order_statistic_filter


# ============================================================
# ALGORITHM: Alpha-Trimmed Mean Filter
# Aim: Restore images corrupted by both Gaussian and impulse noise
# ============================================================
def alpha_trimmed_mean_filter(image: np.ndarray, ksize: int = 3, d: int = 2) -> np.ndarray:
    """
    Alpha-Trimmed Mean Filter:
    Sorts pixels in the neighborhood, trims d/2 lowest and d/2 highest
    pixels, and computes the arithmetic average of the remaining (mn - d) pixels.
    """
    from scipy.ndimage import generic_filter
    ksize = max(3, ksize if ksize % 2 != 0 else ksize + 1)
    d = min(d, ksize * ksize - 1)

    def _trimmed_mean(window):
        sorted_win = np.sort(window)
        trimmed = sorted_win[d // 2 : len(sorted_win) - (d // 2)]
        return np.mean(trimmed) if len(trimmed) > 0 else window[len(window) // 2]

    if image.ndim == 2:
        return generic_filter(image, _trimmed_mean, size=ksize).astype(np.uint8)

    out = np.zeros_like(image)
    for c in range(image.shape[2]):
        out[:, :, c] = generic_filter(image[:, :, c], _trimmed_mean, size=ksize).astype(np.uint8)
    return out


# ============================================================
# CREATIVE STUDIO ALGORITHM: Gaussian Blur
# Aim: Classical bell-curve spatial smoothing
# Formula: G(x, y) = (1 / (2*pi*sigma^2)) * exp( -(x^2+y^2)/(2*sigma^2) )
# ============================================================
def gaussian_blur(image: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Gaussian Spatial Low-Pass Smoothing.
    """
    ksize = max(3, ksize if ksize % 2 != 0 else ksize + 1)
    sigma = max(0.1, float(sigma))
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


# ============================================================
# CREATIVE STUDIO ALGORITHM: Directional Motion Blur
# Aim: Simulate linear camera movement or high-speed subject travel
# ============================================================
def motion_blur(image: np.ndarray, size: int = 15, angle: float = 0.0) -> np.ndarray:
    """
    Directional Motion Blur:
    Constructs a linear line kernel oriented at angle degrees.
    """
    size = max(3, int(size))
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2

    rad = np.radians(angle)
    dx = np.cos(rad) * (size / 2.0)
    dy = np.sin(rad) * (size / 2.0)

    pt1 = (int(center - dx), int(center - dy))
    pt2 = (int(center + dx), int(center + dy))
    cv2.line(kernel, pt1, pt2, 1.0, 1)

    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel /= kernel_sum

    return cv2.filter2D(image, -1, kernel)


# ============================================================
# CREATIVE STUDIO ALGORITHM: Bilateral Filter (Smart Blur / Skin Smoother)
# Aim: Smooth flat color variations while strictly preserving sharp edges
# Formula: Combines spatial Gaussian distance with radiometric intensity Gaussian
# ============================================================
def bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    """
    Photoshop Smart Blur / Retouching:
    Blurs noise and skin blemishes while keeping high-contrast edges intact.
    """
    return cv2.bilateralFilter(image, d=int(d), sigmaColor=float(sigma_color), sigmaSpace=float(sigma_space))


# ============================================================
# CREATIVE STUDIO ALGORITHM: Photoshop Unsharp Mask (USM)
# Aim: The gold standard professional edge sharpening filter
# Formula: Sharpened = Original + Amount * (Original - Blurred)
# ============================================================
def unsharp_mask(image: np.ndarray, amount: float = 1.5, radius: int = 3, threshold: int = 0) -> np.ndarray:
    """
    Photoshop Unsharp Masking:
    - amount: Scaling multiplier for the high-frequency edge mask
    - radius: Gaussian blur radius
    - threshold: Minimum brightness difference required to sharpen
    """
    radius = max(3, radius if radius % 2 != 0 else radius + 1)
    blurred = cv2.GaussianBlur(image, (radius, radius), 0)

    diff = image.astype(np.float32) - blurred.astype(np.float32)

    if threshold > 0:
        mask = np.abs(diff) >= float(threshold)
        diff = diff * mask

    sharpened = image.astype(np.float32) + float(amount) * diff
    return np.clip(sharpened, 0, 255).astype(np.uint8)

