"""
Artistic filters and creative effects.
"""

import cv2
import numpy as np


def pencil_sketch(image: np.ndarray, ksize: int = 21, color_mode: str = "pencil_gray") -> np.ndarray:
    """Generate hand-drawn pencil sketch using color dodge blend."""
    ksize = max(3, ksize if ksize % 2 != 0 else ksize + 1)

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (ksize, ksize), 0)
    dodge = cv2.divide(gray, 255 - blurred, scale=256)
    sketch_gray = np.clip(dodge, 0, 255).astype(np.uint8)

    if color_mode == "pencil_color" and image.ndim == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = sketch_gray
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if image.ndim == 3 and color_mode == "pencil_gray":
        return cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2RGB)

    return sketch_gray


def cartoonify_filter(
    image: np.ndarray,
    num_bilateral: int = 3,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
    edge_block_size: int = 9
) -> np.ndarray:
    """Cartoon cel-shading with bilateral smoothing and edge outline overlay."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)

    edge_block_size = max(3, edge_block_size if edge_block_size % 2 != 0 else edge_block_size + 1)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        edge_block_size, 2
    )

    color_flats = image.copy()
    for _ in range(max(1, num_bilateral)):
        color_flats = cv2.bilateralFilter(color_flats, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return cv2.bitwise_and(color_flats, edges_rgb)


def emboss_filter(image: np.ndarray, azimuth: float = 45.0, depth: float = 1.0) -> np.ndarray:
    """Directional bas-relief 3D emboss convolution."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    rad = np.radians(azimuth)
    dx = np.cos(rad) * depth
    dy = np.sin(rad) * depth

    kernel = np.array([
        [-dx - dy, -dy, 0],
        [-dx,      1.0, dx],
        [0,        dy,  dx + dy]
    ], dtype=np.float32)

    embossed = cv2.filter2D(gray.astype(np.float32), -1, kernel) + 128.0
    embossed_clipped = np.clip(embossed, 0, 255).astype(np.uint8)

    if image.ndim == 3:
        return cv2.cvtColor(embossed_clipped, cv2.COLOR_GRAY2RGB)
    return embossed_clipped


def vignette_filter(image: np.ndarray, radius: float = 0.75, intensity: float = 0.6) -> np.ndarray:
    """Radial optical vignette lens falloff shading."""
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    y, x = np.ogrid[:h, :w]
    max_dist = np.sqrt(cx ** 2 + cy ** 2) * max(0.1, radius)
    dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    norm_dist = np.clip(dist_from_center / max_dist, 0.0, 1.0)
    falloff = 1.0 - (intensity * (norm_dist ** 2))
    falloff = np.clip(falloff, 0.0, 1.0)

    img_f = image.astype(np.float32)
    if image.ndim == 3:
        vignetted = img_f * falloff[:, :, None]
    else:
        vignetted = img_f * falloff

    return np.clip(vignetted, 0, 255).astype(np.uint8)


def tilt_shift_filter(
    image: np.ndarray,
    focus_y: float = 0.5,
    focus_height: float = 0.25,
    blur_amount: int = 15
) -> np.ndarray:
    """Tilt-shift miniature focus effect with gradient blur."""
    h, w = image.shape[:2]
    blur_amount = max(3, blur_amount if blur_amount % 2 != 0 else blur_amount + 1)

    blurred = cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)

    y = np.linspace(0, 1, h)
    center_y = np.clip(focus_y, 0.0, 1.0)
    half_h = max(0.05, focus_height / 2.0)

    dist = np.abs(y - center_y)
    alpha = 1.0 - np.clip((dist - half_h) / (half_h * 1.5), 0.0, 1.0)
    alpha = np.tile(alpha[:, None], (1, w))

    img_f = image.astype(np.float32)
    blur_f = blurred.astype(np.float32)

    if image.ndim == 3:
        out = img_f * alpha[:, :, None] + blur_f * (1.0 - alpha[:, :, None])
    else:
        out = img_f * alpha + blur_f * (1.0 - alpha)

    return np.clip(out, 0, 255).astype(np.uint8)
