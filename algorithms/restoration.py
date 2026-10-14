"""Image restoration, stochastic noise injection, and spatial filtering algorithms."""

import cv2
import numpy as np


def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, sigma: float = 25.0) -> np.ndarray:
    """Adds zero-mean Gaussian electronic sensor noise."""
    noise = np.random.normal(mean, sigma, image.shape)
    degraded = image.astype(np.float32) + noise
    return np.clip(degraded, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(image: np.ndarray, salt_prob: float = 0.02, pepper_prob: float = 0.02) -> np.ndarray:
    """Randomly injects pure white (salt) and pure black (pepper) impulse pixels."""
    noisy = image.copy()
    h, w = image.shape[:2]
    total_pixels = h * w

    num_salt = int(total_pixels * salt_prob)
    if num_salt > 0:
        ys = np.random.randint(0, h, num_salt)
        xs = np.random.randint(0, w, num_salt)
        noisy[ys, xs] = 255

    num_pepper = int(total_pixels * pepper_prob)
    if num_pepper > 0:
        yp = np.random.randint(0, h, num_pepper)
        xp = np.random.randint(0, w, num_pepper)
        noisy[yp, xp] = 0

    return noisy


def add_speckle_noise(image: np.ndarray, variance: float = 0.04) -> np.ndarray:
    """Adds multiplicative speckle noise characteristic of coherent wave imaging."""
    std = np.sqrt(variance)
    noise = np.random.normal(0, std, image.shape)
    degraded = image.astype(np.float32) + (image.astype(np.float32) * noise)
    return np.clip(degraded, 0, 255).astype(np.uint8)


def arithmetic_mean_filter(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Smooths variations using an unweighted sliding box window average."""
    ksize = max(3, ksize if ksize % 2 != 0 else ksize + 1)
    return cv2.blur(image, (ksize, ksize))


def median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Removes impulse noise while preserving sharp edge boundaries."""
    ksize = max(3, ksize if ksize % 2 != 0 else ksize + 1)
    return cv2.medianBlur(image, ksize)


def geometric_mean_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Logarithmic averaging filter that preserves fine edge sharpness."""
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


def contraharmonic_mean_filter(image: np.ndarray, ksize: int = 3, Q: float = 1.5) -> np.ndarray:
    """Order-Q filter: positive Q eliminates pepper noise, negative Q eliminates salt noise."""
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


def order_statistic_filter(image: np.ndarray, ksize: int = 3, mode: str = "median") -> np.ndarray:
    """Non-linear order statistic filtering (min, max, or midpoint)."""
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
    return cv2.medianBlur(image, ksize)


order_statistic_min_max_filter = order_statistic_filter


def alpha_trimmed_mean_filter(image: np.ndarray, ksize: int = 3, d: int = 2) -> np.ndarray:
    """Trims extreme pixel values in the window and averages the remainder."""
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


def gaussian_blur(image: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Applies isotropic Gaussian spatial smoothing."""
    ksize = max(3, ksize if ksize % 2 != 0 else ksize + 1)
    sigma = max(0.1, float(sigma))
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def motion_blur(image: np.ndarray, size: int = 15, angle: float = 0.0) -> np.ndarray:
    """Constructs a linear line kernel oriented at angle degrees to simulate camera shake."""
    size = max(3, int(size))
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2

    rad = np.radians(angle)
    dx = np.cos(rad) * (size / 2.0)
    dy = np.sin(rad) * (size / 2.0)

    cv2.line(kernel, (int(center - dx), int(center - dy)), (int(center + dx), int(center + dy)), 1.0, 1)
    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel /= kernel_sum
    return cv2.filter2D(image, -1, kernel)


def bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    """Edge-preserving bilateral smoothing for skin retouching and noise reduction."""
    return cv2.bilateralFilter(image, d=int(d), sigmaColor=float(sigma_color), sigmaSpace=float(sigma_space))


def unsharp_mask(image: np.ndarray, amount: float = 1.5, radius: int = 3, threshold: int = 0) -> np.ndarray:
    """Sharpens high-frequency edge details using blurred difference mask."""
    radius = max(3, radius if radius % 2 != 0 else radius + 1)
    blurred = cv2.GaussianBlur(image, (radius, radius), 0)
    diff = image.astype(np.float32) - blurred.astype(np.float32)

    if threshold > 0:
        diff = diff * (np.abs(diff) >= float(threshold))

    sharpened = image.astype(np.float32) + float(amount) * diff
    return np.clip(sharpened, 0, 255).astype(np.uint8)
