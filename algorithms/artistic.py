"""
=============================================================================
ImageLab Pro - Creative FX & Photoshop Studio Mechanics
Professional Creative Filters & Artistic Content Creation Tools
=============================================================================
Covers:
- Hand-Drawn Pencil Sketch (Color Dodge Blend + Inverted Gaussian Blur)
- Cel-Shaded Cartoonify (Bilateral Smoothing + Adaptive Edge Overlay)
- 3D Directional Emboss / Bas-Relief Convolution
- Radial Optical Vignette Falloff Shading
- Tilt-Shift Miniature Depth-of-Field Blur
=============================================================================
"""

import cv2
import numpy as np


# ============================================================
# CREATIVE STUDIO ALGORITHM: Hand-Drawn Pencil Sketch
# Aim: Replicate physical graphite pencil sketching
# Technique: Grayscale conversion + Inverted Gaussian Blur + Color Dodge Blend
# Formula: Sketch(x, y) = min(255, (Gray * 256) / (255 - Blur + 1))
# ============================================================
def pencil_sketch(image: np.ndarray, ksize: int = 21, color_mode: str = "pencil_gray") -> np.ndarray:
    """
    Generates authentic hand-drawn pencil sketch art.
    - 'pencil_gray': Classical monochromatic graphite sketch
    - 'pencil_color': Colored pencil illustration
    """
    ksize = max(3, ksize if ksize % 2 != 0 else ksize + 1)

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Step 1: Invert grayscale image
    inverted = 255 - gray

    # Step 2: Apply wide Gaussian blur to inverted frame
    blurred = cv2.GaussianBlur(inverted, (ksize, ksize), 0)

    # Step 3: Color Dodge blend: Gray / (255 - Blurred)
    dodge = cv2.divide(gray, 255 - blurred, scale=256)
    sketch_gray = np.clip(dodge, 0, 255).astype(np.uint8)

    if color_mode == "pencil_color" and image.ndim == 3:
        # Blend sketch shading with original color
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = sketch_gray
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if image.ndim == 3 and color_mode == "pencil_gray":
        return cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2RGB)

    return sketch_gray


# ============================================================
# CREATIVE STUDIO ALGORITHM: Cel-Shaded Cartoonify
# Aim: Transform photographic scenes into animated comic/cartoon art
# Technique: Multi-pass Bilateral filter + Adaptive edge contouring
# ============================================================
def cartoonify_filter(
    image: np.ndarray,
    num_bilateral: int = 3,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
    edge_block_size: int = 9
) -> np.ndarray:
    """
    Cel-Shading Cartoon Filter:
    1. Repeated bilateral filtering flattens color textures while preserving edges.
    2. Adaptive thresholding extracts clean black line-art ink outlines.
    3. Bitwise AND merges colors with black ink contours.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Step 1: Extract edge outline map using median blur and adaptive threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)

    edge_block_size = max(3, edge_block_size if edge_block_size % 2 != 0 else edge_block_size + 1)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        edge_block_size, 2
    )

    # Step 2: Apply multi-pass bilateral filtering for smooth comic-book color flats
    color_flats = image.copy()
    for _ in range(max(1, num_bilateral)):
        color_flats = cv2.bilateralFilter(color_flats, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    # Step 3: Overlay black ink lines onto color flats
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cartoon = cv2.bitwise_and(color_flats, edges_rgb)

    return cartoon


# ============================================================
# CREATIVE STUDIO ALGORITHM: 3D Directional Emboss (Bas-Relief)
# Aim: Replicate 3D tactile stone or metal surface relief
# Technique: Directional convolution difference mask + 128 gray pedestal
# Kernel: [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]
# ============================================================
def emboss_filter(image: np.ndarray, azimuth: float = 45.0, depth: float = 1.0) -> np.ndarray:
    """
    Embossing Filter:
    Convolves image with a directional edge-gradient operator and offsets
    by 128 to simulate a metallic/stone bas-relief illuminated at an angle.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    rad = np.radians(azimuth)
    dx = np.cos(rad) * depth
    dy = np.sin(rad) * depth

    # Directional relief convolution kernel
    kernel = np.array([
        [-dx - dy, -dy, 0],
        [-dx,      1.0, dx],
        [0,        dy,  dx + dy]
    ], dtype=np.float32)

    # Convolve and add 128 neutral pedestal
    embossed = cv2.filter2D(gray.astype(np.float32), -1, kernel) + 128.0
    embossed_clipped = np.clip(embossed, 0, 255).astype(np.uint8)

    if image.ndim == 3:
        return cv2.cvtColor(embossed_clipped, cv2.COLOR_GRAY2RGB)
    return embossed_clipped


# ============================================================
# CREATIVE STUDIO ALGORITHM: Radial Optical Vignette Falloff
# Aim: Simulate optical lens falloff shading and cinematic spotlighting
# Formula: V(r) = 1 - intensity * (r / max_r)^2
# ============================================================
def vignette_filter(image: np.ndarray, radius: float = 0.75, intensity: float = 0.6) -> np.ndarray:
    """
    Radial Vignette Effect:
    Dims the outer corners and periphery while preserving brightness in the center.
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Step 1: Create normalized Euclidean coordinate grid from center
    y, x = np.ogrid[:h, :w]
    max_dist = np.sqrt(cx ** 2 + cy ** 2) * max(0.1, radius)
    dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Step 2: Compute smooth cosine / quadratic falloff mask [0, 1]
    norm_dist = np.clip(dist_from_center / max_dist, 0.0, 1.0)
    falloff = 1.0 - (intensity * (norm_dist ** 2))
    falloff = np.clip(falloff, 0.0, 1.0)

    # Step 3: Multiply image channels with falloff mask
    img_f = image.astype(np.float32)
    if image.ndim == 3:
        vignetted = img_f * falloff[:, :, None]
    else:
        vignetted = img_f * falloff

    return np.clip(vignetted, 0, 255).astype(np.uint8)


# ============================================================
# CREATIVE STUDIO ALGORITHM: Tilt-Shift Miniature Blur
# Aim: Replicate miniature model photography via shallow depth-of-field
# Technique: Horizontal in-focus band with smooth Gaussian blur gradient
# ============================================================
def tilt_shift_filter(
    image: np.ndarray,
    focus_y: float = 0.5,
    focus_height: float = 0.25,
    blur_amount: int = 15
) -> np.ndarray:
    """
    Tilt-Shift Miniature Effect:
    Blurs the top and bottom regions with increasing intensity while keeping
    a central horizontal band in sharp focus, mimicking a tilt-shift camera lens.
    """
    h, w = image.shape[:2]
    blur_amount = max(3, blur_amount if blur_amount % 2 != 0 else blur_amount + 1)

    # Step 1: Compute fully blurred image
    blurred = cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)

    # Step 2: Create linear vertical alpha mask [0 = blurred, 1 = sharp]
    y = np.linspace(0, 1, h)
    center_y = np.clip(focus_y, 0.0, 1.0)
    half_h = max(0.05, focus_height / 2.0)

    # Distance from focus center
    dist = np.abs(y - center_y)
    alpha = 1.0 - np.clip((dist - half_h) / (half_h * 1.5), 0.0, 1.0)
    alpha = np.tile(alpha[:, None], (1, w))

    # Step 3: Alpha blend sharp original with blurred copy
    img_f = image.astype(np.float32)
    blur_f = blurred.astype(np.float32)

    if image.ndim == 3:
        out = img_f * alpha[:, :, None] + blur_f * (1.0 - alpha[:, :, None])
    else:
        out = img_f * alpha + blur_f * (1.0 - alpha)

    return np.clip(out, 0, 255).astype(np.uint8)
