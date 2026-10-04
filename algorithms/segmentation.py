"""
=============================================================================
ImageLab Pro - Feature Segmentation & Edge Analysis
Aligned with Academic DIP Lab Experiments (Exp 09, Exp 10)
=============================================================================
Covers:
- Sobel Gradient Edge Detection (Exp 09 Cell 2)
- Prewitt Differential Edge Detection (Exp 09 Cell 3)
- Laplacian Second-Derivative Operator (Exp 09 Cell 4)
- Canny Hysteresis Edge Detector (Exp 09)
- High-Boost Spatial Sharpening
- Global Manual Thresholding
- Otsu Automatic Optimal Thresholding
- Iterative Optimal Thresholding
- Parametric Hough Line Transform (Exp 10)
=============================================================================
"""

import cv2
import numpy as np


# ============================================================
# DIP EXPERIMENT REF: EXP 09 - CELL 2 (SOBEL OPERATOR)
# Algorithm: Sobel Edge Detection & Gradient Balancer
# Aim: Calculate intensity gradients in X and Y to highlight boundaries
# Formula:
# Gx = Sobel(image, dx=1, dy=0), Gy = Sobel(image, dx=0, dy=1)
# Magnitude = sqrt( alpha * Gx^2 + beta * Gy^2 )
# ============================================================
def directional_gradient_sobel(
    image: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    ksize: int = 3
) -> np.ndarray:
    """
    Sobel Edge Detector:
    Computes first-order spatial gradients along X (vertical edges)
    and Y (horizontal edges), then combines them into gradient magnitude.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    ksize = 3 if ksize not in [1, 3, 5, 7] else ksize

    # Step 1: Calculate gradient in X direction (dx=1, dy=0)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)

    # Step 2: Calculate gradient in Y direction (dx=0, dy=1)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    # Step 3: Compute weighted gradient magnitude
    mag = np.sqrt(alpha * (gx ** 2) + beta * (gy ** 2))

    return np.clip(mag, 0, 255).astype(np.uint8)


# ============================================================
# DIP EXPERIMENT REF: EXP 09 - CELL 3 (PREWITT OPERATOR)
# Algorithm: Prewitt Differential Gradient
# Aim: Detect horizontal and vertical edges using 3x3 difference masks
# Kernels:
# Kx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
# Ky = [[-1, -1, -1], [ 0,  0,  0], [ 1,  1,  1]]
# ============================================================
def prewitt_gradient(image: np.ndarray) -> np.ndarray:
    """
    Prewitt Edge Detection Operator:
    Convolves image with horizontal (Kx) and vertical (Ky) difference masks.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Step 1: Define Prewitt horizontal and vertical convolution masks
    kx = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32)

    ky = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=np.float32)

    # Step 2: Filter image using 2D convolution
    gx = cv2.filter2D(gray.astype(np.float32), -1, kx)
    gy = cv2.filter2D(gray.astype(np.float32), -1, ky)

    # Step 3: Calculate gradient magnitude: sqrt(gx^2 + gy^2)
    mag = np.hypot(gx, gy)

    return np.clip(mag, 0, 255).astype(np.uint8)


# ============================================================
# DIP EXPERIMENT REF: EXP 09 - CELL 4 (LAPLACIAN OPERATOR)
# Algorithm: Laplacian Second-Derivative Operator
# Aim: Detect zero-crossings and sharp edges equally in all directions
# Formula: del^2 f = d^2 f / dx^2 + d^2 f / dy^2
# ============================================================
def laplacian_second_derivative(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Laplacian Second-Order Differential Operator:
    Isotropic operator highlighting rapid intensity transitions.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    ksize = 3 if ksize not in [1, 3, 5] else ksize

    # Step 1: Compute second derivatives using Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)

    # Step 2: Take absolute value and clip to [0, 255]
    return np.clip(np.abs(lap), 0, 255).astype(np.uint8)


# ============================================================
# ALGORITHM: Parametric High-Boost Sharpening Filter
# Aim: Amplify high-frequency edges while retaining original base tones
# Formula: f_hb = A * f(x, y) - f_blurred(x, y)
# ============================================================
def parametric_high_boost(image: np.ndarray, ksize: int = 5, boost_a: float = 1.5) -> np.ndarray:
    """
    High-Boost Filter:
    - boost_a = 1.0: Standard Unsharp Masking
    - boost_a > 1.0: High-Boost Filtering (preserves background intensity)
    """
    ksize = max(3, ksize if ksize % 2 != 0 else ksize + 1)

    # Step 1: Smooth image using Gaussian blur
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0).astype(np.float32)
    img_f = image.astype(np.float32)

    # Step 2: Synthesize high-boost sharpened frame
    hb = boost_a * img_f - blurred

    return np.clip(hb, 0, 255).astype(np.uint8)


# ============================================================
# DIP EXPERIMENT REF: EXP 09 - CANNY EDGE DETECTOR
# Algorithm: Canny Hysteresis Edge Detector
# Aim: Extract thin, continuous, clean single-pixel wide edge maps
# ============================================================
def canny_edge_detector(
    image: np.ndarray,
    threshold_low: int = 50,
    threshold_high: int = 150,
    aperture_size: int = 3
) -> np.ndarray:
    """
    Canny Edge Detector:
    1. Gaussian smoothing
    2. Gradient calculation (Sobel)
    3. Non-maximum suppression (thinning)
    4. Hysteresis thresholding (linking strong and weak edges)
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    aperture = 3 if aperture_size not in [3, 5, 7] else aperture_size
    edges = cv2.Canny(gray, threshold_low, threshold_high, apertureSize=aperture)

    return edges


# ============================================================
# ALGORITHM: Global Manual Thresholding
# Aim: Convert grayscale into binary {0, 255} based on threshold T
# Formula: g(x, y) = 255 if f(x, y) > T else 0
# ============================================================
def manual_threshold(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    """
    Global Fixed-Cutoff Binarization.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


# Backward-compatible alias
global_manual_threshold = manual_threshold


# ============================================================
# ALGORITHM: Otsu Automatic Optimal Binarization
# Aim: Automatically determine optimal threshold by maximizing between-class variance
# Formula: sigma_B^2(T) = w0(T) * w1(T) * (mu0(T) - mu1(T))^2
# ============================================================
def otsu_global_threshold(image: np.ndarray) -> np.ndarray:
    """
    Otsu Binarization:
    Computes optimal global threshold automatically from the histogram.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


# Backward-compatible alias
otsu_automatic_threshold = otsu_global_threshold


# ============================================================
# ALGORITHM: Iterative Optimal Thresholding
# Aim: Converge on optimal threshold by balancing background and foreground means
# Formula: T_{k+1} = 0.5 * ( mu_background + mu_foreground )
# ============================================================
def iterative_optimal_threshold(
    image: np.ndarray,
    delta_t: float = 0.5,
    max_iters: int = 50,
    **kwargs
) -> np.ndarray:
    """
    Iterative Threshold Selection:
    Initializes T as image mean, partitions pixels into two groups,
    computes group means, and updates T until change is less than delta_t.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Step 1: Initialize threshold as global image mean
    t_curr = float(np.mean(gray))

    # Step 2: Iterate until convergence
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


# ============================================================
# DIP EXPERIMENT REF: EXP 10 - HOUGH TRANSFORM
# Algorithm: Hough Transform Line Detection
# Aim: Map edge pixels to (rho, theta) parameter space to detect lines
# Formula: rho = x * cos(theta) + y * sin(theta)
# ============================================================
def hough_line_transform(
    image: np.ndarray,
    threshold_votes: int = 80,
    min_line_len: int = 40,
    max_line_gap: int = 10,
    draw_on_source: bool = True,
    **kwargs
) -> np.ndarray:
    """
    Probabilistic Hough Line Transform:
    Detects straight collinear line segments in an edge image.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        canvas = image.copy() if draw_on_source else np.zeros_like(image)
    else:
        gray = image.copy()
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if draw_on_source else np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Step 1: Compute Canny edge map
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Step 2: Vote in (rho, theta) accumulator using HoughLinesP
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=threshold_votes,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap
    )

    # Step 3: Draw detected vector line segments in bright neon cyan
    if lines is not None:
        for line in lines:
            pts = line[0] if (hasattr(line, '__len__') and len(line) == 1 and hasattr(line[0], '__len__')) else line
            if hasattr(pts, '__len__') and len(pts) == 4:
                x1, y1, x2, y2 = pts
                cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (6, 182, 212), 2, cv2.LINE_AA)

    return canvas


# ============================================================
# CREATIVE STUDIO ALGORITHM: Adaptive Local Window Thresholding
# Aim: Binarize images with non-uniform, uneven background lighting
# Formula: T(x, y) = Mean(block) - C or Gaussian(block) - C
# ============================================================
def adaptive_threshold(
    image: np.ndarray,
    block_size: int = 11,
    c: int = 2,
    method: str = "gaussian"
) -> np.ndarray:
    """
    Adaptive Thresholding:
    Calculates dynamic threshold per pixel from its local neighborhood.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    block_size = max(3, block_size if block_size % 2 != 0 else block_size + 1)
    ad_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C

    return cv2.adaptiveThreshold(gray, 255, ad_method, cv2.THRESH_BINARY, block_size, c)


# ============================================================
# DIP EXPERIMENT REF: EXP 10 - PART 2 (HOUGH CIRCLES)
# Algorithm: Hough Circle Transform
# Aim: Detect circular shapes, coins, and pupils via gradient voting
# Formula: (x - a)^2 + (y - b)^2 = r^2
# ============================================================
def hough_circle_transform(
    image: np.ndarray,
    min_dist: int = 30,
    param1: int = 100,
    param2: int = 30,
    min_radius: int = 10,
    max_radius: int = 120,
    draw_on_source: bool = True
) -> np.ndarray:
    """
    Hough Circle Detection:
    Finds circular objects using 2D gradient orientation voting.
    """
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
            # Draw circle boundary in emerald green
            cv2.circle(canvas, center, radius, (16, 185, 129), 2, cv2.LINE_AA)
            # Draw circle center in amber
            cv2.circle(canvas, center, 2, (245, 158, 11), 3, cv2.LINE_AA)

    return canvas


# ============================================================
# CREATIVE STUDIO ALGORITHM: Harris & Stephens Corner Detection
# Aim: Detect geometric corner keypoints invariant to rotation
# Formula: R = det(M) - k * (trace(M))^2
# ============================================================
def harris_corners(
    image: np.ndarray,
    block_size: int = 2,
    ksize: int = 3,
    k: float = 0.04,
    threshold: float = 0.01,
    draw_on_source: bool = True
) -> np.ndarray:
    """
    Harris Corner Detector:
    Highlights corner feature points where intensity changes along all directions.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        canvas = image.copy() if draw_on_source else np.zeros_like(image)
    else:
        gray = image.copy()
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if draw_on_source else np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    gray_f = np.float32(gray)
    dst = cv2.cornerHarris(gray_f, blockSize=block_size, ksize=ksize, k=float(k))
    dst_norm = cv2.normalize(dst, None, 0, 1.0, cv2.NORM_MINMAX)

    # Mark corners in bright rose pink
    corner_mask = dst_norm > threshold
    canvas[corner_mask] = [244, 63, 94]

    return canvas

