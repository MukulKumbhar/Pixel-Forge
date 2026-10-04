"""
=============================================================================
ImageLab Pro - Chromatic Engine & Colorimetry
Aligned with Academic DIP Lab Experiments (Exp 08)
=============================================================================
Covers:
- RGB to HSI Trigonometric Conversion (Exp 08 Parts 1–5)
- Inverse HSI to RGB Reconstruction (Exp 08)
- HSI Vector Color Grading (Hue Rotation, Saturation, Intensity Offset)
- Subtractive CMY/CMYK Separation
- von Kries White Balance & Chromatic Adaptation
- Three-Way Color Grading (Lift, Gamma, Gain)
- Cinematic Color Tone Curve
=============================================================================
"""

import cv2
import numpy as np


# ============================================================
# DIP EXPERIMENT REF: EXP 08 - PARTS 1 TO 5
# Algorithm: RGB to HSI Trigonometric Conversion
# Aim: Convert RGB color image to Hue, Saturation, and Intensity
# Formula:
# I = (R + G + B) / 3
# S = 1 - (3 / (R + G + B)) * min(R, G, B)
# theta = arccos( 0.5 * ((R-G) + (R-B)) / sqrt((R-G)^2 + (R-B)*(G-B)) )
# H = theta if B <= G else (360 - theta)
# ============================================================
def rgb_to_hsi_trigonometric(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard Gonzales & Woods Trigonometric RGB -> HSI transformation:
    - Input: RGB image in [0, 255]
    - Returns: H in [0, 360), S in [0, 1], I in [0, 1]
    """
    # Step 1: Normalize RGB values to [0, 1] range
    img_f = image.astype(np.float64) / 255.0
    r = img_f[:, :, 0]
    g = img_f[:, :, 1]
    b = img_f[:, :, 2]
    eps = 1e-6

    # Step 2: Calculate Intensity (I)
    i = (r + g + b) / 3.0

    # Step 3: Calculate Saturation (S)
    rgb_min = np.minimum(np.minimum(r, g), b)
    sum_rgb = r + g + b
    s = 1.0 - (3.0 / (sum_rgb + eps)) * rgb_min
    s[sum_rgb < eps] = 0.0
    s = np.clip(s, 0.0, 1.0)

    # Step 4: Calculate Hue (H) using analytical trigonometric formula
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + eps
    val = np.clip(num / den, -1.0, 1.0)
    theta = np.degrees(np.arccos(val))

    h = theta.copy()
    h[b > g] = 360.0 - theta[b > g]
    h[s < eps] = 0.0

    return h, s, i


# ============================================================
# DIP EXPERIMENT REF: EXP 08 - INVERSE CONVERSION
# Algorithm: HSI to RGB Sector-Based Reconstruction
# Aim: Reconstruct RGB image from Hue, Saturation, and Intensity
# ============================================================
def hsi_to_rgb_trigonometric(h: np.ndarray, s: np.ndarray, i: np.ndarray) -> np.ndarray:
    """
    Inverse HSI -> RGB Sector-based trigonometric reconstruction:
    Sector 1: 0 <= H < 120 (RG sector)
    Sector 2: 120 <= H < 240 (GB sector)
    Sector 3: 240 <= H <= 360 (BR sector)
    """
    h_rad = np.radians(h % 360.0)
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    # Sector 1: RG sector (0 <= H < 120)
    idx1 = (h >= 0.0) & (h < 120.0)
    b[idx1] = i[idx1] * (1.0 - s[idx1])
    r[idx1] = i[idx1] * (1.0 + (s[idx1] * np.cos(h_rad[idx1])) / (np.cos(np.radians(60.0) - h_rad[idx1]) + 1e-6))
    g[idx1] = 3.0 * i[idx1] - (r[idx1] + b[idx1])

    # Sector 2: GB sector (120 <= H < 240)
    idx2 = (h >= 120.0) & (h < 240.0)
    h_sub2 = h_rad[idx2] - np.radians(120.0)
    r[idx2] = i[idx2] * (1.0 - s[idx2])
    g[idx2] = i[idx2] * (1.0 + (s[idx2] * np.cos(h_sub2)) / (np.cos(np.radians(60.0) - h_sub2) + 1e-6))
    b[idx2] = 3.0 * i[idx2] - (r[idx2] + g[idx2])

    # Sector 3: BR sector (240 <= H <= 360)
    idx3 = (h >= 240.0) & (h <= 360.0)
    h_sub3 = h_rad[idx3] - np.radians(240.0)
    g[idx3] = i[idx3] * (1.0 - s[idx3])
    b[idx3] = i[idx3] * (1.0 + (s[idx3] * np.cos(h_sub3)) / (np.cos(np.radians(60.0) - h_sub3) + 1e-6))
    r[idx3] = 3.0 * i[idx3] - (g[idx3] + b[idx3])

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


# ============================================================
# ALGORITHM: HSI Vector Color Grading
# Aim: Rotate hue angle, scale saturation, and offset intensity
# ============================================================
def hsi_vector_grading(
    image: np.ndarray,
    hue_shift: float = 0.0,
    sat_scale: float = 1.0,
    intensity_offset: float = 0.0
) -> np.ndarray:
    """
    HSI Vector Studio:
    - hue_shift: Angular rotation in degrees [-180, +180]
    - sat_scale: Saturation multiplier [0.0, 3.0]
    - intensity_offset: Intensity offset [-1.0, +1.0]
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    h, s, i = rgb_to_hsi_trigonometric(image)

    # Step 1: Rotate hue angle
    h_mod = (h + hue_shift) % 360.0

    # Step 2: Scale saturation
    s_mod = np.clip(s * sat_scale, 0.0, 1.0)

    # Step 3: Shift intensity
    i_mod = np.clip(i + intensity_offset, 0.0, 1.0)

    # Step 4: Reconstruct RGB image
    return hsi_to_rgb_trigonometric(h_mod, s_mod, i_mod)


# ============================================================
# ALGORITHM: Subtractive CMY / CMYK Separation
# Aim: Convert additive RGB channels into subtractive printing inks
# Formula: C = 1 - R, M = 1 - G, Y = 1 - B, K = min(C, M, Y)
# ============================================================
def cmyk_decomposition(
    image: np.ndarray,
    channel_view: str = "composite_cmy",
    mode: str = None,
    **kwargs
) -> np.ndarray:
    """
    Decomposes RGB into subtractive color printing channels (Cyan, Magenta, Yellow, Key/Black).
    """
    selected_mode = mode or channel_view
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    img_f = image.astype(np.float32) / 255.0
    r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]

    # Subtractive primary inks
    c = 1.0 - r
    m = 1.0 - g
    y = 1.0 - b

    # Black (K) key channel
    k = np.minimum(np.minimum(c, m), y)
    eps = 1e-6
    c_k = np.clip((c - k) / (1.0 - k + eps), 0, 1)
    m_k = np.clip((m - k) / (1.0 - k + eps), 0, 1)
    y_k = np.clip((y - k) / (1.0 - k + eps), 0, 1)

    if selected_mode == "cyan":
        return (c_k * 255.0).astype(np.uint8)
    elif selected_mode == "magenta":
        return (m_k * 255.0).astype(np.uint8)
    elif selected_mode == "yellow":
        return (y_k * 255.0).astype(np.uint8)
    elif selected_mode == "black_k":
        return (k * 255.0).astype(np.uint8)
    else:
        # Standard CMY composite
        cmy = np.stack([c, m, y], axis=-1)
        return (cmy * 255.0).astype(np.uint8)


# Backward-compatible alias
rgb_to_cmyk_visualization = cmyk_decomposition


# ============================================================
# ALGORITHM: Three-Way Color Grading (Lift, Gamma, Gain)
# Aim: Adjust shadows, midtones, and highlights independently
# ============================================================
def three_way_color_grade(
    image: np.ndarray,
    lift: float = 0.0,
    gamma: float = 1.0,
    gain: float = 1.0
) -> np.ndarray:
    """
    Industry Standard 3-Way Lift / Gamma / Gain Split Toning:
    - Lift: Adjusts black level / shadows
    - Gain: Scales white level / highlights
    - Gamma: Adjusts midtones via power-law curve
    Formula: out = [ Gain * (img + Lift * (1 - img)) ]^(1 / Gamma)
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    img_f = image.astype(np.float32) / 255.0

    # Step 1: Apply shadow Lift
    lifted = img_f + lift * (1.0 - img_f)
    lifted = np.clip(lifted, 0.0, 1.0)

    # Step 2: Apply highlight Gain
    gained = lifted * gain
    gained = np.clip(gained, 0.0, 1.0)

    # Step 3: Apply midtone Gamma
    gamma = max(0.1, float(gamma))
    out = np.power(gained, 1.0 / gamma)

    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


# Backward-compatible alias
three_way_lift_gamma_gain = three_way_color_grade


# ============================================================
# ALGORITHM: von Kries White Balance
# Aim: Correct color casts using chromatic adaptation coefficients
# ============================================================
def von_kries_white_balance(image: np.ndarray, temperature: float = 0.0, tint: float = 0.0) -> np.ndarray:
    """
    von Kries Chromatic Adaptation:
    - temperature: Warm amber (+temp) vs Cool blue (-temp)
    - tint: Magenta (+tint) vs Green (-tint)
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    k_r = 1.0 + 0.35 * temperature - 0.15 * tint
    k_g = 1.0 + 0.20 * tint
    k_b = 1.0 - 0.35 * temperature + 0.15 * tint

    k_r = max(0.1, k_r)
    k_g = max(0.1, k_g)
    k_b = max(0.1, k_b)

    img_f = image.astype(np.float32)
    img_f[:, :, 0] *= k_r
    img_f[:, :, 1] *= k_g
    img_f[:, :, 2] *= k_b

    return np.clip(img_f, 0, 255).astype(np.uint8)


# ============================================================
# ALGORITHM: Multi-Channel RGB Matrix Balancer
# Aim: Scale and shift individual Red, Green, and Blue channels
# ============================================================
def multi_channel_rgb_matrix(
    image: np.ndarray,
    r_gain: float = 1.0,
    g_gain: float = 1.0,
    b_gain: float = 1.0,
    r_offset: int = 0,
    g_offset: int = 0,
    b_offset: int = 0
) -> np.ndarray:
    """
    Independent RGB Multi-Channel Offsets & Contrast:
    R' = r_gain * R + r_offset
    G' = g_gain * G + g_offset
    B' = b_gain * B + b_offset
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    img_f = image.astype(np.float32)
    img_f[:, :, 0] = img_f[:, :, 0] * r_gain + r_offset
    img_f[:, :, 1] = img_f[:, :, 1] * g_gain + g_offset
    img_f[:, :, 2] = img_f[:, :, 2] * b_gain + b_offset

    return np.clip(img_f, 0, 255).astype(np.uint8)


# ============================================================
# ALGORITHM: Cinematic Color Tone Curve
# Aim: Apply popular photographic mood profiles (Teal & Orange, Film, Bleach Bypass)
# ============================================================
def apply_3d_lut(image: np.ndarray, profile: str = "teal_orange", strength: float = 1.0) -> np.ndarray:
    """
    Cinematic Color Tone Curves:
    - 'teal_orange': Cool teal shadows, warm golden highlights
    - 'kodak_film': Vintage golden film look with lifted blacks
    - 'bleach_bypass': Silver retention high-contrast desaturated look
    - 'cross_process': S-curve with cyan shadows and saturated yellow-greens
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    img_f = image.astype(np.float32) / 255.0
    r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
    strength = float(np.clip(strength, 0.0, 1.0))

    if profile == "teal_orange":
        # Push shadows toward teal (cyan/blue) and highlights toward warm orange
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        r_lut = r + (1.0 - luma) * (-0.15) + luma * 0.22
        g_lut = g + (1.0 - luma) * (0.05) + luma * 0.05
        b_lut = b + (1.0 - luma) * (0.25) + luma * (-0.20)

    elif profile == "kodak_film":
        # Warm golden tones, lifted blacks, soft contrast
        r_lut = np.power(r, 0.90) * 1.05 + 0.02
        g_lut = np.power(g, 0.95) * 1.00 + 0.01
        b_lut = np.power(b, 1.10) * 0.92

    elif profile == "bleach_bypass":
        # High contrast, desaturated silver tones
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        overlay = np.where(luma < 0.5, 2 * img_f * luma[:, :, None], 1 - 2 * (1 - img_f) * (1 - luma[:, :, None]))
        desat = 0.65 * overlay + 0.35 * luma[:, :, None]
        r_lut, g_lut, b_lut = desat[:, :, 0], desat[:, :, 1], desat[:, :, 2]

    elif profile == "cross_process":
        # High-saturation S-curves with color shifts
        r_lut = 0.5 * (1 + np.sin(np.pi * (r - 0.5)))
        g_lut = np.power(g, 0.85)
        b_lut = np.power(b, 1.25) + 0.08 * (1.0 - b)
    else:
        r_lut, g_lut, b_lut = r, g, b

    # Blend graded look with original according to strength
    r_out = (1.0 - strength) * r + strength * r_lut
    g_out = (1.0 - strength) * g + strength * g_lut
    b_out = (1.0 - strength) * b + strength * b_lut

    graded = np.stack([r_out, g_out, b_out], axis=-1)
    return np.clip(graded * 255.0, 0, 255).astype(np.uint8)


# ============================================================
# CREATIVE STUDIO ALGORITHM: Vibrance & Smart Saturation
# Aim: Replicate Photoshop Vibrance (boosts muted tones, protects skin)
# ============================================================
def vibrance_saturation(image: np.ndarray, vibrance: float = 0.5, saturation: float = 0.0) -> np.ndarray:
    """
    Photoshop Smart Vibrance:
    - vibrance: Selectively saturates less-saturated colors more than saturated ones
    - saturation: Uniform linear saturation scale
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    s = hsv[:, :, 1]

    # Non-linear vibrance curve: boosts low saturation pixels most
    norm_s = s / 255.0
    vibrance_boost = (1.0 - norm_s) * float(vibrance) * 255.0
    s_new = s + vibrance_boost + float(saturation) * s
    hsv[:, :, 1] = np.clip(s_new, 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


# ============================================================
# CREATIVE STUDIO ALGORITHM: Antique Sepia Tint
# Aim: Replicate classic vintage 19th-century silver bromide photography
# Matrix: [0.393, 0.769, 0.189; 0.349, 0.686, 0.168; 0.272, 0.534, 0.131]
# ============================================================
def sepia_vintage(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Classic Antique Sepia Tone Matrix Transform.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ], dtype=np.float32)

    sepia_img = cv2.transform(image, sepia_matrix)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)

    strength = float(np.clip(strength, 0.0, 1.0))
    return cv2.addWeighted(sepia_img, strength, image, 1.0 - strength, 0)


# ============================================================
# CREATIVE STUDIO ALGORITHM: DaVinci Resolve Split Toning
# Aim: Color grade shadows with a cool tint and highlights with a warm tint
# ============================================================
def split_toning(
    image: np.ndarray,
    shadow_hue: float = 210.0,
    shadow_sat: float = 0.35,
    highlight_hue: float = 40.0,
    highlight_sat: float = 0.35
) -> np.ndarray:
    """
    Split Toning (Shadow Tint + Highlight Tint).
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    img_f = image.astype(np.float32) / 255.0
    luma = 0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 2]

    # Shadows weight: (1 - luma), Highlights weight: luma
    s_weight = (1.0 - luma)[:, :, None] * float(shadow_sat)
    h_weight = luma[:, :, None] * float(highlight_sat)

    # Convert hues to RGB tint vectors
    def _hue_to_rgb(h):
        h_rad = np.radians(h % 360.0)
        return np.array([0.5 + 0.5 * np.cos(h_rad), 0.5 + 0.5 * np.cos(h_rad - 2.094), 0.5 + 0.5 * np.cos(h_rad + 2.094)], dtype=np.float32)

    s_rgb = _hue_to_rgb(shadow_hue)
    h_rgb = _hue_to_rgb(highlight_hue)

    graded = img_f + s_weight * (s_rgb - 0.5) + h_weight * (h_rgb - 0.5)
    return np.clip(graded * 255.0, 0, 255).astype(np.uint8)

