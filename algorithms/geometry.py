"""
=============================================================================
ImageLab Pro - Geometric & Spatial Foundations
Aligned with Academic DIP Lab Experiments (Exp 01, 02, 03)
=============================================================================
Covers:
- Spatial Sampling & Discretization (Exp 03: Downsampling & Interpolation)
- Bit-Depth Amplitude Quantization (Exp 03: 8-bit to k-bit Reduction)
- Affine Geometric Transformations (Rotation, Scale, Translation, Shear)
- Canvas Cropping & Padding
=============================================================================
"""

import cv2
import numpy as np


# ============================================================
# DIP EXPERIMENT REF: EXP 03 - PART A (SAMPLING)
# Algorithm: Spatial Downsampling & Upsampling
# Aim: Convert a continuous/high-res image into discrete pixels
#      and observe aliasing and pixelation effects.
# ============================================================
def spatial_downsample_upsample(
    image: np.ndarray,
    scale_factor: float = 0.5,
    interpolation: str = "nearest"
) -> np.ndarray:
    """
    Simulates spatial discretization by downsampling and upsampling back
    to original dimensions to demonstrate pixelation and reconstruction filters.
    """
    h, w = image.shape[:2]

    # Map interpolation algorithm choice
    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA
    }
    interp_flag = interp_map.get(interpolation.lower(), cv2.INTER_NEAREST)

    # Step 1: Calculate reduced dimensions based on scale factor
    target_w = max(4, int(w * scale_factor))
    target_h = max(4, int(h * scale_factor))

    # Step 2: Downsample image (simulates lower spatial sampling rate)
    downscaled = cv2.resize(image, (target_w, target_h), interpolation=interp_flag)

    # Step 3: Upsample back to canvas size to observe visual artifacts
    reconstructed = cv2.resize(downscaled, (w, h), interpolation=interp_flag)

    return reconstructed


# ============================================================
# DIP EXPERIMENT REF: EXP 03 - PART B (QUANTIZATION)
# Algorithm: Bit-Depth Amplitude Quantization
# Aim: Reduce continuous intensity levels into discrete digital steps (k-bits)
# Formula: q = floor(r / (256 / 2^k)) * (255 / (2^k - 1))
# ============================================================
def bit_depth_quantization(image: np.ndarray, bits: int = 4) -> np.ndarray:
    """
    Reduces the intensity resolution from 8-bit [0-255] down to k-bits (1 to 7).
    Demonstrates 'false contouring' where smooth shading turns into visible bands.
    """
    bits = int(np.clip(bits, 1, 8))
    if bits == 8:
        return image.copy()

    # Step 1: Compute total discrete levels available (2^k) and step interval
    levels = 2 ** bits
    step = 256.0 / levels

    # Step 2: Discretize pixels into level index [0, levels - 1]
    quantized = np.floor(image.astype(np.float32) / step)
    quantized = np.clip(quantized, 0, levels - 1)

    # Step 3: Scale discrete levels back to full 8-bit dynamic range [0, 255]
    reconstructed = np.round(quantized * (255.0 / (levels - 1))).astype(np.uint8)

    return reconstructed


# ============================================================
# ALGORITHM: Composite 2D Affine Transformation
# Aim: Apply geometric translation, rotation, scaling, and shearing
# Formula: [x', y', 1]^T = M_affine * [x, y, 1]^T
# ============================================================
def affine_transform(
    image: np.ndarray,
    angle: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    trans_x: float = 0.0,
    trans_y: float = 0.0,
    shear_x: float = 0.0,
    shear_y: float = 0.0
) -> np.ndarray:
    """
    Applies unified 2D Affine matrix transformation about the image center.
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Step 1: Shift image center to origin (0, 0)
    T_center_inv = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)

    # Step 2: Compute rotation matrix (angle in degrees)
    rad = np.radians(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    R = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)

    # Step 3: Compute scaling matrix
    S = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float32)

    # Step 4: Compute shearing matrix
    Sh = np.array([[1, shear_x, 0], [shear_y, 1, 0], [0, 0, 1]], dtype=np.float32)

    # Step 5: Shift origin back to center + apply user translation offsets
    T_center = np.array([[1, 0, cx + trans_x], [0, 1, cy + trans_y], [0, 0, 1]], dtype=np.float32)

    # Step 6: Combine transformations into a single 3x3 matrix
    M_composite = T_center @ Sh @ S @ R @ T_center_inv
    M_2x3 = M_composite[:2, :]

    # Step 7: Apply affine warp with bilinear interpolation
    transformed = cv2.warpAffine(
        image, M_2x3, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return transformed


# ============================================================
# ALGORITHM: Canvas Resize & Letterbox Fitting
# Aim: Scale image to fit target resolution while preserving aspect ratio.
# ============================================================
def canvas_resize(image: np.ndarray, width: int, height: int, mode: str = "fit") -> np.ndarray:
    """
    Resizes image canvas with aspect ratio preservation (fit with black padding or stretch).
    """
    h, w = image.shape[:2]
    if width <= 0 or height <= 0:
        return image.copy()

    if mode == "stretch":
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    # Step 1: Compute uniform scale factor to fit inside target bounding box
    scale = min(width / w, height / h)
    nw, nh = int(w * scale), int(h * scale)

    # Step 2: Resize image using area or linear interpolation
    scaled = cv2.resize(
        image, (nw, nh),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    )

    # Step 3: Create blank target canvas and paste centered
    if image.ndim == 3:
        canvas = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    else:
        canvas = np.zeros((height, width), dtype=image.dtype)

    dx = (width - nw) // 2
    dy = (height - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = scaled

    return canvas


# ============================================================
# CREATIVE STUDIO ALGORITHM: Optical Lens Distortion Correction
# Aim: Simulate or correct radial barrel and pincushion optical distortion
# Formula: r_distorted = r * (1 + k1 * r^2 + k2 * r^4)
# ============================================================
def lens_distortion(image: np.ndarray, k1: float = -0.15, k2: float = 0.0) -> np.ndarray:
    """
    Radial Lens Distortion:
    - k1 < 0: Barrel distortion (wide-angle / action camera fisheye look)
    - k1 > 0: Pincushion distortion (telephoto lens compression)
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    fx = fy = max(w, h)
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([k1, k2, 0, 0], dtype=np.float32)

    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    return undistorted


# ============================================================
# CREATIVE STUDIO ALGORITHM: Perspective Keystone Correction
# Aim: Correct architectural keystoning or tilted document perspective
# ============================================================
def perspective_keystone(image: np.ndarray, tilt_x: float = 0.1, tilt_y: float = 0.0) -> np.ndarray:
    """
    Perspective Homography Transform:
    Warp quad corners to simulate perspective tilt or straighten keystone distortions.
    """
    h, w = image.shape[:2]
    dx = w * tilt_x * 0.25
    dy = h * tilt_y * 0.25

    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([
        [max(0.0, dx), max(0.0, dy)],
        [min(float(w), w - dx), max(0.0, -dy)],
        [min(float(w), w + dx), min(float(h), h - dy)],
        [max(0.0, -dx), min(float(h), h + dy)]
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

