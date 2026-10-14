"""
Morphological mechanics and topological image filtering.
"""

import cv2
import numpy as np


def get_structuring_element(shape: str = "rect", ksize: int = 3) -> np.ndarray:
    """Generate structuring element kernel for morphological operations."""
    ksize = max(3, ksize if ksize % 2 != 0 else ksize + 1)
    shape_map = {
        "rect": cv2.MORPH_RECT,
        "cross": cv2.MORPH_CROSS,
        "ellipse": cv2.MORPH_ELLIPSE
    }
    morph_shape = shape_map.get(shape.lower(), cv2.MORPH_RECT)
    return cv2.getStructuringElement(morph_shape, (ksize, ksize))


def morphological_dilate(image: np.ndarray, shape: str = "rect", ksize: int = 3, iterations: int = 1) -> np.ndarray:
    """Expand foreground regions using morphological dilation."""
    se = get_structuring_element(shape, ksize)
    return cv2.dilate(image, se, iterations=iterations)


def morphological_erode(image: np.ndarray, shape: str = "rect", ksize: int = 3, iterations: int = 1) -> np.ndarray:
    """Shrink foreground regions and remove small noise using erosion."""
    se = get_structuring_element(shape, ksize)
    return cv2.erode(image, se, iterations=iterations)


def morphological_open(image: np.ndarray, shape: str = "rect", ksize: int = 3) -> np.ndarray:
    """Erosion followed by dilation to remove small bright noise."""
    se = get_structuring_element(shape, ksize)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, se)


def morphological_close(image: np.ndarray, shape: str = "rect", ksize: int = 3) -> np.ndarray:
    """Dilation followed by erosion to fill small dark holes and gaps."""
    se = get_structuring_element(shape, ksize)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)


def morphological_gradient(image: np.ndarray, shape: str = "rect", ksize: int = 3) -> np.ndarray:
    """Difference between dilation and erosion to outline object boundaries."""
    se = get_structuring_element(shape, ksize)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, se)


def morphological_tophat(image: np.ndarray, shape: str = "rect", ksize: int = 9) -> np.ndarray:
    """White top-hat transform to isolate bright elements smaller than the kernel."""
    se = get_structuring_element(shape, ksize)
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, se)


def morphological_blackhat(image: np.ndarray, shape: str = "rect", ksize: int = 9) -> np.ndarray:
    """Black-hat transform to isolate dark elements smaller than the kernel."""
    se = get_structuring_element(shape, ksize)
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, se)


def boundary_extraction(image: np.ndarray, shape: str = "rect", ksize: int = 3) -> np.ndarray:
    """Extract object boundaries by subtracting eroded mask from original."""
    se = get_structuring_element(shape, ksize)
    eroded = cv2.erode(image, se)
    return cv2.subtract(image, eroded)


def hit_and_miss_transform(image: np.ndarray, target: str = "isolated_points") -> np.ndarray:
    """Hit-or-miss transform to detect specific pixel configurations."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    if target == "isolated_points":
        kernel = np.array([
            [0, -1, 0],
            [-1, 1, -1],
            [0, -1, 0]
        ], dtype=np.int32)
    else:
        kernel = np.array([
            [-1, -1, -1],
            [-1,  1,  1],
            [-1,  1,  0]
        ], dtype=np.int32)

    return cv2.morphologyEx(binary, cv2.MORPH_HITMISS, kernel)
