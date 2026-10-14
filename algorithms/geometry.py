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


# ============================================================
# CREATIVE STUDIO ALGORITHM: Vector & Multi-Shape Cropper
# Aim: Crop image using various shapes (Circle, Square, Rectangle,
#      Ellipse, Triangle, Diamond, Hexagon, Octagon, Star, Heart,
#      Polygon, Custom Bounding Box) with subpixel anti-aliasing,
#      Gaussian feathering, alpha transparency, and tight-bounds crop.
# ============================================================
def vector_shape_crop(
    image: np.ndarray,
    shape: str = "circle",
    center_x: float = 0.5,
    center_y: float = 0.5,
    size: float = 0.8,
    aspect_ratio: str = "1:1",
    rotation: float = 0.0,
    feather: int = 0,
    bg_fill: str = "transparent",
    tight_crop: bool = True,
    sides: int = 5,
    star_inner_ratio: float = 0.382
) -> np.ndarray:
    """
    Crops canvas using analytical vector shapes:
    - Circle: Avatar & profile circular crop
    - Square: 1:1 square crop
    - Rectangle: Aspect-ratio locked or freeform box
    - Ellipse: Smooth 2D ellipse with rotation
    - Triangle: Equilateral / Isosceles polygon
    - Diamond: Rhombus / 45-degree diamond
    - Hexagon / Octagon: 6 or 8-sided regular polygon
    - Star: 5-point (or custom) star badge
    - Heart: Smooth mathematical cardioid heart curve
    - Polygon: Regular N-sided polygon (3 to 16 sides)
    - Custom Bounding Box: Freeform coordinate crop

    Parameters:
        image: Source RGB or Grayscale ndarray
        shape: Shape name string
        center_x: Normalized horizontal center (0.0 to 1.0)
        center_y: Normalized vertical center (0.0 to 1.0)
        size: Normalized size scale (0.05 to 1.0)
        aspect_ratio: Aspect ratio preset ("1:1", "4:3", "16:9", "3:2", "9:16", "freeform")
        rotation: Rotation angle in degrees (-180 to 180)
        feather: Gaussian soft-edge feathering radius in pixels (0 to 50)
        bg_fill: Background mode ("transparent", "black", "white", "studio_dark")
        tight_crop: Whether to crop canvas to shape bounding box (True) or keep canvas size (False)
        sides: Number of sides for regular polygon (3 to 16)
        star_inner_ratio: Ratio of inner to outer radius for star (0.1 to 0.9)
    """
    h, w = image.shape[:2]
    cx = float(np.clip(center_x, 0.0, 1.0)) * w
    cy = float(np.clip(center_y, 0.0, 1.0)) * h
    base_dim = min(w, h)
    s = max(8.0, float(np.clip(size, 0.05, 1.5)) * base_dim * 0.5)

    # Resolve aspect ratio multiplier
    ar_map = {
        "1:1": 1.0,
        "4:3": 4.0 / 3.0,
        "16:9": 16.0 / 9.0,
        "3:2": 3.0 / 2.0,
        "2:1": 2.0,
        "9:16": 9.0 / 16.0,
        "3:4": 3.0 / 4.0,
        "freeform": 1.0
    }
    ar_val = ar_map.get(aspect_ratio.lower(), 1.0)

    # Initialize empty single-channel binary/float mask
    mask = np.zeros((h, w), dtype=np.uint8)
    shape_norm = shape.lower().strip()

    rad_rot = np.radians(rotation)
    cos_r = np.cos(rad_rot)
    sin_r = np.sin(rad_rot)

    def _rotate_points(pts: np.ndarray, pivot_x: float, pivot_y: float) -> np.ndarray:
        shifted = pts - np.array([pivot_x, pivot_y])
        rotated_x = shifted[:, 0] * cos_r - shifted[:, 1] * sin_r + pivot_x
        rotated_y = shifted[:, 0] * sin_r + shifted[:, 1] * cos_r + pivot_y
        return np.column_stack([rotated_x, rotated_y])

    # ---------------------------------------------------------
    # 1. SHAPE VECTOR GEOMETRY RENDERING
    # ---------------------------------------------------------
    if shape_norm in ["circle", "round"]:
        cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(s)), 255, -1, lineType=cv2.LINE_AA)

    elif shape_norm in ["square"]:
        half = s
        box_pts = np.array([
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half]
        ], dtype=np.float32)
        rot_pts = _rotate_points(box_pts, cx, cy)
        cv2.fillPoly(mask, [np.int32(np.round(rot_pts))], 255, lineType=cv2.LINE_AA)

    elif shape_norm in ["rectangle", "rect"]:
        half_w = s * np.sqrt(ar_val)
        half_h = s / np.sqrt(ar_val)
        box_pts = np.array([
            [cx - half_w, cy - half_h],
            [cx + half_w, cy - half_h],
            [cx + half_w, cy + half_h],
            [cx - half_w, cy + half_h]
        ], dtype=np.float32)
        rot_pts = _rotate_points(box_pts, cx, cy)
        cv2.fillPoly(mask, [np.int32(np.round(rot_pts))], 255, lineType=cv2.LINE_AA)

    elif shape_norm in ["ellipse", "oval"]:
        rx = int(round(s * np.sqrt(ar_val)))
        ry = int(round(s / np.sqrt(ar_val)))
        cv2.ellipse(mask, (int(round(cx)), int(round(cy))), (max(2, rx), max(2, ry)), rotation, 0, 360, 255, -1, lineType=cv2.LINE_AA)

    elif shape_norm in ["triangle"]:
        pts = []
        for i in range(3):
            angle = -np.pi / 2.0 + (2.0 * np.pi * i / 3.0)
            pts.append([cx + s * np.cos(angle), cy + s * np.sin(angle)])
        rot_pts = _rotate_points(np.array(pts, dtype=np.float32), cx, cy)
        cv2.fillPoly(mask, [np.int32(np.round(rot_pts))], 255, lineType=cv2.LINE_AA)

    elif shape_norm in ["diamond", "rhombus"]:
        rx = s * np.sqrt(ar_val)
        ry = s / np.sqrt(ar_val)
        pts = np.array([
            [cx, cy - ry],
            [cx + rx, cy],
            [cx, cy + ry],
            [cx - rx, cy]
        ], dtype=np.float32)
        rot_pts = _rotate_points(pts, cx, cy)
        cv2.fillPoly(mask, [np.int32(np.round(rot_pts))], 255, lineType=cv2.LINE_AA)

    elif shape_norm in ["hexagon"]:
        pts = []
        for i in range(6):
            angle = -np.pi / 2.0 + (2.0 * np.pi * i / 6.0)
            pts.append([cx + s * np.cos(angle), cy + s * np.sin(angle)])
        rot_pts = _rotate_points(np.array(pts, dtype=np.float32), cx, cy)
        cv2.fillPoly(mask, [np.int32(np.round(rot_pts))], 255, lineType=cv2.LINE_AA)

    elif shape_norm in ["octagon"]:
        pts = []
        for i in range(8):
            angle = -np.pi / 2.0 + (2.0 * np.pi * i / 8.0)
            pts.append([cx + s * np.cos(angle), cy + s * np.sin(angle)])
        rot_pts = _rotate_points(np.array(pts, dtype=np.float32), cx, cy)
        cv2.fillPoly(mask, [np.int32(np.round(rot_pts))], 255, lineType=cv2.LINE_AA)

    elif shape_norm in ["star"]:
        points_count = max(3, int(sides)) if sides > 3 else 5
        r_outer = s
        r_inner = s * float(np.clip(star_inner_ratio, 0.1, 0.9))
        pts = []
        total_verts = points_count * 2
        for i in range(total_verts):
            r = r_outer if (i % 2 == 0) else r_inner
            angle = -np.pi / 2.0 + (np.pi * i / points_count)
            pts.append([cx + r * np.cos(angle), cy + r * np.sin(angle)])
        rot_pts = _rotate_points(np.array(pts, dtype=np.float32), cx, cy)
        cv2.fillPoly(mask, [np.int32(np.round(rot_pts))], 255, lineType=cv2.LINE_AA)

    elif shape_norm in ["heart", "love"]:
        t = np.linspace(0, 2.0 * np.pi, 360)
        # Smooth parametric cardioid heart equations
        hx = 16.0 * (np.sin(t) ** 3)
        hy = -(13.0 * np.cos(t) - 5.0 * np.cos(2.0 * t) - 2.0 * np.cos(3.0 * t) - np.cos(4.0 * t))
        # Normalize to [-1.0, 1.0]
        hx = (hx / 16.0) * s
        hy = (hy / 17.0) * s
        pts = np.column_stack([cx + hx, cy + hy]).astype(np.float32)
        rot_pts = _rotate_points(pts, cx, cy)
        cv2.fillPoly(mask, [np.int32(np.round(rot_pts))], 255, lineType=cv2.LINE_AA)

    elif shape_norm in ["polygon", "ngon"]:
        n_sides = max(3, min(16, int(sides)))
        pts = []
        for i in range(n_sides):
            angle = -np.pi / 2.0 + (2.0 * np.pi * i / float(n_sides))
            pts.append([cx + s * np.cos(angle), cy + s * np.sin(angle)])
        rot_pts = _rotate_points(np.array(pts, dtype=np.float32), cx, cy)
        cv2.fillPoly(mask, [np.int32(np.round(rot_pts))], 255, lineType=cv2.LINE_AA)

    else:
        # Default: Full-circle crop fallback
        cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(s)), 255, -1, lineType=cv2.LINE_AA)

    # ---------------------------------------------------------
    # 2. EDGE FEATHERING (SOFT BOUNDARY BLUR)
    # ---------------------------------------------------------
    feather = int(np.clip(feather, 0, 50))
    if feather > 0:
        ksize = int(feather * 2 + 1) | 1
        mask_float = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), feather / 2.0)
        mask = np.clip(mask_float, 0.0, 255.0).astype(np.uint8)

    # ---------------------------------------------------------
    # 3. BACKGROUND BLENDING & CHANNEL SYNTHESIS
    # ---------------------------------------------------------
    bg_fill_norm = bg_fill.lower().strip()

    if bg_fill_norm in ["transparent", "rgba", "alpha"]:
        if image.ndim == 3 and image.shape[2] == 4:
            # Combine incoming alpha with shape mask
            combined_alpha = np.clip((image[:, :, 3].astype(np.float32) * mask.astype(np.float32)) / 255.0, 0, 255).astype(np.uint8)
            result = np.dstack([image[:, :, :3], combined_alpha])
        elif image.ndim == 3:
            result = np.dstack([image, mask])
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            result = np.dstack([rgb, mask])
    else:
        # Solid background fill color mapping
        bg_colors = {
            "black": np.array([0, 0, 0], dtype=np.float32),
            "white": np.array([255, 255, 255], dtype=np.float32),
            "studio_dark": np.array([22, 27, 34], dtype=np.float32)
        }
        bg_vec = bg_colors.get(bg_fill_norm, np.array([0, 0, 0], dtype=np.float32))

        # Ensure base image is RGB 3-channel for solid compositing
        if image.ndim == 3 and image.shape[2] == 4:
            base_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.ndim == 2:
            base_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            base_rgb = image.copy()

        alpha_f = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
        composite = base_rgb.astype(np.float32) * alpha_f + bg_vec * (1.0 - alpha_f)
        result = np.clip(composite, 0.0, 255.0).astype(np.uint8)

    # ---------------------------------------------------------
    # 4. TIGHT BOUNDING BOX CROP (OPTIONAL CANVAS TRIM)
    # ---------------------------------------------------------
    if tight_crop:
        nonzero_y, nonzero_x = np.nonzero(mask > 2)
        if len(nonzero_y) > 0 and len(nonzero_x) > 0:
            y_min = max(0, int(np.min(nonzero_y)))
            y_max = min(h, int(np.max(nonzero_y)) + 1)
            x_min = max(0, int(np.min(nonzero_x)))
            x_max = min(w, int(np.max(nonzero_x)) + 1)
            if (y_max > y_min) and (x_max > x_min):
                result = result[y_min:y_max, x_min:x_max]

    return result


