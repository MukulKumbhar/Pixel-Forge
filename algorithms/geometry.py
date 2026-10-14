"""Geometric and spatial foundation image processing algorithms."""

import cv2
import numpy as np


def spatial_downsample_upsample(
    image: np.ndarray,
    scale_factor: float = 0.5,
    interpolation: str = "nearest"
) -> np.ndarray:
    """Downsamples and upsamples canvas to simulate resolution discretization."""
    h, w = image.shape[:2]
    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA
    }
    interp_flag = interp_map.get(interpolation.lower(), cv2.INTER_NEAREST)

    target_w = max(4, int(w * scale_factor))
    target_h = max(4, int(h * scale_factor))

    downscaled = cv2.resize(image, (target_w, target_h), interpolation=interp_flag)
    return cv2.resize(downscaled, (w, h), interpolation=interp_flag)


def bit_depth_quantization(image: np.ndarray, bits: int = 4) -> np.ndarray:
    """Reduces intensity resolution from 8-bit to k-bits."""
    bits = int(np.clip(bits, 1, 8))
    if bits == 8:
        return image.copy()

    levels = 2 ** bits
    step = 256.0 / levels
    quantized = np.clip(np.floor(image.astype(np.float32) / step), 0, levels - 1)
    return np.round(quantized * (255.0 / (levels - 1))).astype(np.uint8)


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
    """Applies rotation, scale, translation, and shearing about image center."""
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    T_center_inv = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
    rad = np.radians(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    R = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
    S = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float32)
    Sh = np.array([[1, shear_x, 0], [shear_y, 1, 0], [0, 0, 1]], dtype=np.float32)
    T_center = np.array([[1, 0, cx + trans_x], [0, 1, cy + trans_y], [0, 0, 1]], dtype=np.float32)

    M_composite = T_center @ Sh @ S @ R @ T_center_inv
    M_2x3 = M_composite[:2, :]

    return cv2.warpAffine(
        image, M_2x3, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )


def canvas_resize(image: np.ndarray, width: int, height: int, mode: str = "fit") -> np.ndarray:
    """Resizes canvas with aspect ratio fit padding or stretch."""
    h, w = image.shape[:2]
    if width <= 0 or height <= 0:
        return image.copy()

    if mode == "stretch":
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    scale = min(width / w, height / h)
    nw, nh = int(w * scale), int(h * scale)
    scaled = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

    canvas_shape = (height, width, image.shape[2]) if image.ndim == 3 else (height, width)
    canvas = np.zeros(canvas_shape, dtype=image.dtype)
    dx = (width - nw) // 2
    dy = (height - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = scaled
    return canvas


def lens_distortion(image: np.ndarray, k1: float = -0.15, k2: float = 0.0) -> np.ndarray:
    """Simulates or corrects radial barrel and pincushion optical lens distortion."""
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    fx = fy = max(w, h)
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([k1, k2, 0, 0], dtype=np.float32)

    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    return cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)


def perspective_keystone(image: np.ndarray, tilt_x: float = 0.1, tilt_y: float = 0.0) -> np.ndarray:
    """Warps quad corners to correct or simulate perspective keystone tilt."""
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
    """Crops canvas using analytical vector shapes with anti-aliasing and transparency."""
    h, w = image.shape[:2]
    cx = float(np.clip(center_x, 0.0, 1.0)) * w
    cy = float(np.clip(center_y, 0.0, 1.0)) * h
    base_dim = min(w, h)
    s = max(8.0, float(np.clip(size, 0.05, 1.5)) * base_dim * 0.5)

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
    mask = np.zeros((h, w), dtype=np.uint8)
    shape_norm = shape.lower().strip()

    rad_rot = np.radians(rotation)
    cos_r = np.cos(rad_rot)
    sin_r = np.sin(rad_rot)

    def _rotate_points(pts: np.ndarray, pivot_x: float, pivot_y: float) -> np.ndarray:
        shifted = pts - np.array([pivot_x, pivot_y])
        rx = shifted[:, 0] * cos_r - shifted[:, 1] * sin_r + pivot_x
        ry = shifted[:, 0] * sin_r + shifted[:, 1] * cos_r + pivot_y
        return np.column_stack([rx, ry])

    # Rasterize shape geometry
    if shape_norm in ["circle", "round"]:
        cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(s)), 255, -1, lineType=cv2.LINE_AA)

    elif shape_norm in ["square"]:
        half = s
        box_pts = np.array([
            [cx - half, cy - half], [cx + half, cy - half],
            [cx + half, cy + half], [cx - half, cy + half]
        ], dtype=np.float32)
        cv2.fillPoly(mask, [np.int32(np.round(_rotate_points(box_pts, cx, cy)))], 255, lineType=cv2.LINE_AA)

    elif shape_norm in ["rectangle", "rect"]:
        half_w = s * np.sqrt(ar_val)
        half_h = s / np.sqrt(ar_val)
        box_pts = np.array([
            [cx - half_w, cy - half_h], [cx + half_w, cy - half_h],
            [cx + half_w, cy + half_h], [cx - half_w, cy + half_h]
        ], dtype=np.float32)
        cv2.fillPoly(mask, [np.int32(np.round(_rotate_points(box_pts, cx, cy)))], 255, lineType=cv2.LINE_AA)

    elif shape_norm in ["ellipse", "oval"]:
        rx = int(round(s * np.sqrt(ar_val)))
        ry = int(round(s / np.sqrt(ar_val)))
        cv2.ellipse(mask, (int(round(cx)), int(round(cy))), (max(2, rx), max(2, ry)), rotation, 0, 360, 255, -1, lineType=cv2.LINE_AA)

    elif shape_norm in ["triangle"]:
        pts = [[cx + s * np.cos(-np.pi / 2.0 + 2.0 * np.pi * i / 3.0),
                cy + s * np.sin(-np.pi / 2.0 + 2.0 * np.pi * i / 3.0)] for i in range(3)]
        cv2.fillPoly(mask, [np.int32(np.round(_rotate_points(np.array(pts, dtype=np.float32), cx, cy)))], 255, lineType=cv2.LINE_AA)

    elif shape_norm in ["diamond", "rhombus"]:
        rx, ry = s * np.sqrt(ar_val), s / np.sqrt(ar_val)
        pts = np.array([[cx, cy - ry], [cx + rx, cy], [cx, cy + ry], [cx - rx, cy]], dtype=np.float32)
        cv2.fillPoly(mask, [np.int32(np.round(_rotate_points(pts, cx, cy)))], 255, lineType=cv2.LINE_AA)

    elif shape_norm in ["hexagon"]:
        pts = [[cx + s * np.cos(-np.pi / 2.0 + 2.0 * np.pi * i / 6.0),
                cy + s * np.sin(-np.pi / 2.0 + 2.0 * np.pi * i / 6.0)] for i in range(6)]
        cv2.fillPoly(mask, [np.int32(np.round(_rotate_points(np.array(pts, dtype=np.float32), cx, cy)))], 255, lineType=cv2.LINE_AA)

    elif shape_norm in ["octagon"]:
        pts = [[cx + s * np.cos(-np.pi / 2.0 + 2.0 * np.pi * i / 8.0),
                cy + s * np.sin(-np.pi / 2.0 + 2.0 * np.pi * i / 8.0)] for i in range(8)]
        cv2.fillPoly(mask, [np.int32(np.round(_rotate_points(np.array(pts, dtype=np.float32), cx, cy)))], 255, lineType=cv2.LINE_AA)

    elif shape_norm in ["star"]:
        points_count = max(3, int(sides)) if sides > 3 else 5
        r_outer, r_inner = s, s * float(np.clip(star_inner_ratio, 0.1, 0.9))
        pts = []
        for i in range(points_count * 2):
            r = r_outer if (i % 2 == 0) else r_inner
            angle = -np.pi / 2.0 + (np.pi * i / points_count)
            pts.append([cx + r * np.cos(angle), cy + r * np.sin(angle)])
        cv2.fillPoly(mask, [np.int32(np.round(_rotate_points(np.array(pts, dtype=np.float32), cx, cy)))], 255, lineType=cv2.LINE_AA)

    elif shape_norm in ["heart", "love"]:
        t = np.linspace(0, 2.0 * np.pi, 360)
        hx = (16.0 * (np.sin(t) ** 3) / 16.0) * s
        hy = (-(13.0 * np.cos(t) - 5.0 * np.cos(2.0 * t) - 2.0 * np.cos(3.0 * t) - np.cos(4.0 * t)) / 17.0) * s
        pts = np.column_stack([cx + hx, cy + hy]).astype(np.float32)
        cv2.fillPoly(mask, [np.int32(np.round(_rotate_points(pts, cx, cy)))], 255, lineType=cv2.LINE_AA)

    elif shape_norm in ["polygon", "ngon"]:
        n_sides = max(3, min(16, int(sides)))
        pts = [[cx + s * np.cos(-np.pi / 2.0 + 2.0 * np.pi * i / float(n_sides)),
                cy + s * np.sin(-np.pi / 2.0 + 2.0 * np.pi * i / float(n_sides))] for i in range(n_sides)]
        cv2.fillPoly(mask, [np.int32(np.round(_rotate_points(np.array(pts, dtype=np.float32), cx, cy)))], 255, lineType=cv2.LINE_AA)

    else:
        cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(s)), 255, -1, lineType=cv2.LINE_AA)

    # Edge feathering
    feather = int(np.clip(feather, 0, 50))
    if feather > 0:
        ksize = int(feather * 2 + 1) | 1
        mask = np.clip(cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), feather / 2.0), 0.0, 255.0).astype(np.uint8)

    # Background blending
    bg_fill_norm = bg_fill.lower().strip()
    if bg_fill_norm in ["transparent", "rgba", "alpha"]:
        if image.ndim == 3 and image.shape[2] == 4:
            combined_alpha = np.clip((image[:, :, 3].astype(np.float32) * mask.astype(np.float32)) / 255.0, 0, 255).astype(np.uint8)
            result = np.dstack([image[:, :, :3], combined_alpha])
        elif image.ndim == 3:
            result = np.dstack([image, mask])
        else:
            result = np.dstack([cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), mask])
    else:
        bg_colors = {
            "black": np.array([0, 0, 0], dtype=np.float32),
            "white": np.array([255, 255, 255], dtype=np.float32),
            "studio_dark": np.array([22, 27, 34], dtype=np.float32)
        }
        bg_vec = bg_colors.get(bg_fill_norm, np.array([0, 0, 0], dtype=np.float32))

        if image.ndim == 3 and image.shape[2] == 4:
            base_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.ndim == 2:
            base_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            base_rgb = image.copy()

        alpha_f = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
        composite = base_rgb.astype(np.float32) * alpha_f + bg_vec * (1.0 - alpha_f)
        result = np.clip(composite, 0.0, 255.0).astype(np.uint8)

    # Tight bounding box trimming
    if tight_crop:
        nonzero_y, nonzero_x = np.nonzero(mask > 2)
        if len(nonzero_y) > 0 and len(nonzero_x) > 0:
            y_min, y_max = max(0, int(np.min(nonzero_y))), min(h, int(np.max(nonzero_y)) + 1)
            x_min, x_max = max(0, int(np.min(nonzero_x))), min(w, int(np.max(nonzero_x)) + 1)
            if (y_max > y_min) and (x_max > x_min):
                result = result[y_min:y_max, x_min:x_max]

    return result
