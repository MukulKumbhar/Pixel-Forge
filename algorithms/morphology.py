"""
ImageLab Pro - Morphological Mechanics
Covers Unit 6: Set-theoretic geometry, topological probes (structuring elements),
boundary extraction, compound open/close, morphological gradient, top-hat/black-hat,
and hit-or-miss transformations.
Pure NumPy / OpenCV mathematical implementations.
"""

import cv2
import numpy as np


def get_structuring_element(shape: str = "rect", ksize: int = 3) -> np.ndarray:
    """
    Generates topological probe / structuring element (SE):
    - shape: "rect", "cross", "ellipse"
    - ksize: odd integer (3, 5, 7, 9)
    """
    ksize = max(3, ksize if ksize % 2 != 0 else ksize + 1)
    shape_map = {
        "rect": cv2.MORPH_RECT,
        "cross": cv2.MORPH_CROSS,
        "ellipse": cv2.MORPH_ELLIPSE
    }
    morph_shape = shape_map.get(shape.lower(), cv2.MORPH_RECT)
    return cv2.getStructuringElement(morph_shape, (ksize, ksize))


def morphological_dilate(image: np.ndarray, shape: str = "rect", ksize: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Morphological Dilation: A ⊕ B = { z | (B_hat)_z ∩ A ≠ ∅ }
    Expands connected foreground components and bridges narrow breaks.
    """
    se = get_structuring_element(shape, ksize)
    return cv2.dilate(image, se, iterations=iterations)


def morphological_erode(image: np.ndarray, shape: str = "rect", ksize: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Morphological Erosion: A ⊖ B = { z | (B)_z ⊆ A }
    Shrinks foreground regions and eliminates small isolated noise.
    """
    se = get_structuring_element(shape, ksize)
    return cv2.erode(image, se, iterations=iterations)


def morphological_open(image: np.ndarray, shape: str = "rect", ksize: int = 3) -> np.ndarray:
    """
    Morphological Opening: A ∘ B = (A ⊖ B) ⊕ B
    Erosion followed by dilation. Eliminates small protrusions and isolated bright details
    smaller than the structuring element without noticeably altering overall geometry.
    """
    se = get_structuring_element(shape, ksize)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, se)


def morphological_close(image: np.ndarray, shape: str = "rect", ksize: int = 3) -> np.ndarray:
    """
    Morphological Closing: A • B = (A ⊕ B) ⊖ B
    Dilation followed by erosion. Fills small dark holes, bridges narrow gaps,
    and fuses adjacent disjoint regions.
    """
    se = get_structuring_element(shape, ksize)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)


def morphological_gradient(image: np.ndarray, shape: str = "rect", ksize: int = 3) -> np.ndarray:
    """
    Morphological Gradient: G(A) = (A ⊕ B) - (A ⊖ B)
    Difference between dilation and erosion; highlights topological boundaries.
    """
    se = get_structuring_element(shape, ksize)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, se)


def morphological_tophat(image: np.ndarray, shape: str = "rect", ksize: int = 9) -> np.ndarray:
    """
    Top-Hat (White Top-Hat) Transform: T_hat(A) = A - (A ∘ B)
    Subtracts opened image from original; extracts bright features smaller than SE.
    Useful for nonuniform background illumination correction.
    """
    se = get_structuring_element(shape, ksize)
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, se)


def morphological_blackhat(image: np.ndarray, shape: str = "rect", ksize: int = 9) -> np.ndarray:
    """
    Black-Hat (Bottom-Hat) Transform: B_hat(A) = (A • B) - A
    Subtracts original from closed image; isolates dark structures and depressions.
    """
    se = get_structuring_element(shape, ksize)
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, se)


def boundary_extraction(image: np.ndarray, shape: str = "rect", ksize: int = 3) -> np.ndarray:
    """
    Morphological Boundary Extraction:
    beta(A) = A - (A ⊖ B)
    Subtracts internal eroded core from foreground set A.
    """
    se = get_structuring_element(shape, ksize)
    eroded = cv2.erode(image, se)
    return cv2.subtract(image, eroded)


def hit_and_miss_transform(image: np.ndarray, target: str = "isolated_points") -> np.ndarray:
    """
    Hit-or-Miss Transform: A ⊛ B = (A ⊖ B1) ∩ (A^c ⊖ B2)
    Detects specific topological structures or pixel configurations.
    - target="isolated_points": probes single isolated foreground pixels
    - target="corners": probes top-left corner structures
    """
    # Ensure binary input [0, 255]
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    if target == "isolated_points":
        # Kernel: center is 1 (hit), 4-connected neighbors are -1 (miss), corners 0 (don't care)
        kernel = np.array([
            [0, -1, 0],
            [-1, 1, -1],
            [0, -1, 0]
        ], dtype=np.int32)
    else:  # corners
        # Top-left corner detection kernel
        kernel = np.array([
            [-1, -1, -1],
            [-1,  1,  1],
            [-1,  1,  0]
        ], dtype=np.int32)

    output = cv2.morphologyEx(binary, cv2.MORPH_HITMISS, kernel)
    return output
