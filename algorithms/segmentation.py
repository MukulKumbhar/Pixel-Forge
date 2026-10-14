"""
Feature segmentation and edge analysis algorithms.
"""

import cv2
import numpy as np


def directional_gradient_sobel(
    image: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    ksize: int = 3
) -> np.ndarray:
    """Sobel edge detector combining X and Y gradients."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    ksize = 3 if ksize not in [1, 3, 5, 7] else ksize
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = np.sqrt(alpha * (gx ** 2) + beta * (gy ** 2))
    return np.clip(mag, 0, 255).astype(np.uint8)


def prewitt_gradient(image: np.ndarray) -> np.ndarray:
    """Prewitt edge detection using 3x3 difference masks."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

    gx = cv2.filter2D(gray.astype(np.float32), -1, kx)
    gy = cv2.filter2D(gray.astype(np.float32), -1, ky)
    mag = np.hypot(gx, gy)
    return np.clip(mag, 0, 255).astype(np.uint8)


def laplacian_second_derivative(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Laplacian second-derivative operator for edge transitions."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    ksize = 3 if ksize not in [1, 3, 5] else ksize
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    return np.clip(np.abs(lap), 0, 255).astype(np.uint8)


def parametric_high_boost(image: np.ndarray, ksize: int = 5, boost_a: float = 1.5) -> np.ndarray:
    """High-boost filter to sharpen edges while preserving base tones."""
    ksize = max(3, ksize if ksize % 2 != 0 else ksize + 1)
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0).astype(np.float32)
    img_f = image.astype(np.float32)
    hb = boost_a * img_f - blurred
    return np.clip(hb, 0, 255).astype(np.uint8)


def canny_edge_detector(
    image: np.ndarray,
    threshold_low: int = 50,
    threshold_high: int = 150,
    aperture_size: int = 3
) -> np.ndarray:
    """Canny hysteresis edge detector for clean thin edges."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    aperture = 3 if aperture_size not in [3, 5, 7] else aperture_size
    return cv2.Canny(gray, threshold_low, threshold_high, apertureSize=aperture)


def manual_threshold(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Global fixed cutoff binarization."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


# Backward-compatible alias
global_manual_threshold = manual_threshold


def otsu_global_threshold(image: np.ndarray) -> np.ndarray:
    """Otsu automatic thresholding via histogram variance."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


# Backward-compatible alias
otsu_automatic_threshold = otsu_global_threshold


def iterative_optimal_threshold(
    image: np.ndarray,
    delta_t: float = 0.5,
    max_iters: int = 50,
    **kwargs
) -> np.ndarray:
    """Iterative threshold selection based on background/foreground means."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    t_curr = float(np.mean(gray))
    for _ in range(max_iters):
        g1 = gray[gray > t_curr]
        g2 = gray[gray <= t_curr]

        m1 = float(np.mean(g1)) if len(g1) > 0 else 0.0
        m2 = float(np.mean(g2)) if len(g2) > 0 else 0.0

        t_next = 0.5 * (m1 + m2)
        if abs(t_next - t_curr) < delta_t:
            t_curr = t_next
            break
        t_curr = t_next

    _, binary = cv2.threshold(gray, int(t_curr), 255, cv2.THRESH_BINARY)
    return binary


def hough_line_transform(
    image: np.ndarray,
    threshold_votes: int = 80,
    min_line_len: int = 40,
    max_line_gap: int = 10,
    draw_on_source: bool = True,
    **kwargs
) -> np.ndarray:
    """Probabilistic Hough line transform to detect straight line segments."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        canvas = image.copy() if draw_on_source else np.zeros_like(image)
    else:
        gray = image.copy()
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if draw_on_source else np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=threshold_votes,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap
    )

    if lines is not None:
        for line in lines:
            pts = line[0] if (hasattr(line, '__len__') and len(line) == 1 and hasattr(line[0], '__len__')) else line
            if hasattr(pts, '__len__') and len(pts) == 4:
                x1, y1, x2, y2 = pts
                cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (6, 182, 212), 2, cv2.LINE_AA)

    return canvas


def adaptive_threshold(
    image: np.ndarray,
    block_size: int = 11,
    c: int = 2,
    method: str = "gaussian"
) -> np.ndarray:
    """Adaptive local window thresholding for uneven lighting."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    block_size = max(3, block_size if block_size % 2 != 0 else block_size + 1)
    ad_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C

    return cv2.adaptiveThreshold(gray, 255, ad_method, cv2.THRESH_BINARY, block_size, c)


def hough_circle_transform(
    image: np.ndarray,
    min_dist: int = 30,
    param1: int = 100,
    param2: int = 30,
    min_radius: int = 10,
    max_radius: int = 120,
    draw_on_source: bool = True
) -> np.ndarray:
    """Hough circle transform to locate circular objects."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        canvas = image.copy() if draw_on_source else np.zeros_like(image)
    else:
        gray = image.copy()
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if draw_on_source else np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    gray_blur = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray_blur, cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(1, min_dist),
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            center = (int(c[0]), int(c[1]))
            radius = int(c[2])
            cv2.circle(canvas, center, radius, (16, 185, 129), 2, cv2.LINE_AA)
            cv2.circle(canvas, center, 2, (245, 158, 11), 3, cv2.LINE_AA)

    return canvas


def harris_corners(
    image: np.ndarray,
    block_size: int = 2,
    ksize: int = 3,
    k: float = 0.04,
    threshold: float = 0.01,
    draw_on_source: bool = True
) -> np.ndarray:
    """Harris corner detector for geometric keypoint detection."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        canvas = image.copy() if draw_on_source else np.zeros_like(image)
    else:
        gray = image.copy()
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if draw_on_source else np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    gray_f = np.float32(gray)
    dst = cv2.cornerHarris(gray_f, blockSize=block_size, ksize=ksize, k=float(k))
    dst_norm = cv2.normalize(dst, None, 0, 1.0, cv2.NORM_MINMAX)

    corner_mask = dst_norm > threshold
    canvas[corner_mask] = [244, 63, 94]

    return canvas
