"""Color spaces, chromatic adaptation, and color grading algorithms."""

import cv2
import numpy as np


def rgb_to_hsi_trigonometric(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts RGB image to H (0-360), S (0-1), and I (0-1)."""
    img_f = image.astype(np.float64) / 255.0
    r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
    eps = 1e-6

    i = (r + g + b) / 3.0
    rgb_min = np.minimum(np.minimum(r, g), b)
    sum_rgb = r + g + b
    s = 1.0 - (3.0 / (sum_rgb + eps)) * rgb_min
    s[sum_rgb < eps] = 0.0
    s = np.clip(s, 0.0, 1.0)

    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + eps
    theta = np.degrees(np.arccos(np.clip(num / den, -1.0, 1.0)))

    h = theta.copy()
    h[b > g] = 360.0 - theta[b > g]
    h[s < eps] = 0.0
    return h, s, i


def hsi_to_rgb_trigonometric(h: np.ndarray, s: np.ndarray, i: np.ndarray) -> np.ndarray:
    """Reconstructs RGB from HSI sectors (RG, GB, BR)."""
    h_rad = np.radians(h % 360.0)
    r, g, b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)

    # RG sector (0 <= H < 120)
    idx1 = (h >= 0.0) & (h < 120.0)
    b[idx1] = i[idx1] * (1.0 - s[idx1])
    r[idx1] = i[idx1] * (1.0 + (s[idx1] * np.cos(h_rad[idx1])) / (np.cos(np.radians(60.0) - h_rad[idx1]) + 1e-6))
    g[idx1] = 3.0 * i[idx1] - (r[idx1] + b[idx1])

    # GB sector (120 <= H < 240)
    idx2 = (h >= 120.0) & (h < 240.0)
    h_sub2 = h_rad[idx2] - np.radians(120.0)
    r[idx2] = i[idx2] * (1.0 - s[idx2])
    g[idx2] = i[idx2] * (1.0 + (s[idx2] * np.cos(h_sub2)) / (np.cos(np.radians(60.0) - h_sub2) + 1e-6))
    b[idx2] = 3.0 * i[idx2] - (r[idx2] + g[idx2])

    # BR sector (240 <= H <= 360)
    idx3 = (h >= 240.0) & (h <= 360.0)
    h_sub3 = h_rad[idx3] - np.radians(240.0)
    g[idx3] = i[idx3] * (1.0 - s[idx3])
    b[idx3] = i[idx3] * (1.0 + (s[idx3] * np.cos(h_sub3)) / (np.cos(np.radians(60.0) - h_sub3) + 1e-6))
    r[idx3] = 3.0 * i[idx3] - (g[idx3] + b[idx3])

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def hsi_vector_grading(
    image: np.ndarray,
    hue_shift: float = 0.0,
    sat_scale: float = 1.0,
    intensity_offset: float = 0.0
) -> np.ndarray:
    """Adjusts hue rotation, saturation scaling, and intensity offset in HSI space."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    h, s, i = rgb_to_hsi_trigonometric(image)
    h_mod = (h + hue_shift) % 360.0
    s_mod = np.clip(s * sat_scale, 0.0, 1.0)
    i_mod = np.clip(i + intensity_offset, 0.0, 1.0)
    return hsi_to_rgb_trigonometric(h_mod, s_mod, i_mod)


def cmyk_decomposition(
    image: np.ndarray,
    channel_view: str = "composite_cmy",
    mode: str = None,
    **kwargs
) -> np.ndarray:
    """Decomposes RGB into subtractive CMYK printing channels."""
    selected_mode = mode or channel_view
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    img_f = image.astype(np.float32) / 255.0
    r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]

    c, m, y = 1.0 - r, 1.0 - g, 1.0 - b
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
    return (np.stack([c, m, y], axis=-1) * 255.0).astype(np.uint8)


rgb_to_cmyk_visualization = cmyk_decomposition


def three_way_color_grade(
    image: np.ndarray,
    lift: float = 0.0,
    gamma: float = 1.0,
    gain: float = 1.0
) -> np.ndarray:
    """Three-way color wheel grading: shadow lift, midtone gamma, and highlight gain."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    img_f = image.astype(np.float32) / 255.0
    lifted = np.clip(img_f + lift * (1.0 - img_f), 0.0, 1.0)
    gained = np.clip(lifted * gain, 0.0, 1.0)
    gamma = max(0.1, float(gamma))
    out = np.power(gained, 1.0 / gamma)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


three_way_lift_gamma_gain = three_way_color_grade


def von_kries_white_balance(image: np.ndarray, temperature: float = 0.0, tint: float = 0.0) -> np.ndarray:
    """Adjusts color temperature (warm/cool) and tint (magenta/green)."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    k_r = max(0.1, 1.0 + 0.35 * temperature - 0.15 * tint)
    k_g = max(0.1, 1.0 + 0.20 * tint)
    k_b = max(0.1, 1.0 - 0.35 * temperature + 0.15 * tint)

    img_f = image.astype(np.float32)
    img_f[:, :, 0] *= k_r
    img_f[:, :, 1] *= k_g
    img_f[:, :, 2] *= k_b
    return np.clip(img_f, 0, 255).astype(np.uint8)


def multi_channel_rgb_matrix(
    image: np.ndarray,
    r_gain: float = 1.0,
    g_gain: float = 1.0,
    b_gain: float = 1.0,
    r_offset: int = 0,
    g_offset: int = 0,
    b_offset: int = 0
) -> np.ndarray:
    """Independent RGB channel gain and offset mixer."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    img_f = image.astype(np.float32)
    img_f[:, :, 0] = img_f[:, :, 0] * r_gain + r_offset
    img_f[:, :, 1] = img_f[:, :, 1] * g_gain + g_offset
    img_f[:, :, 2] = img_f[:, :, 2] * b_gain + b_offset
    return np.clip(img_f, 0, 255).astype(np.uint8)


def apply_3d_lut(image: np.ndarray, profile: str = "teal_orange", strength: float = 1.0) -> np.ndarray:
    """Applies film tone curves (Teal & Orange, Kodak Film, Bleach Bypass, Cross Process)."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    img_f = image.astype(np.float32) / 255.0
    r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
    strength = float(np.clip(strength, 0.0, 1.0))

    if profile == "teal_orange":
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        r_lut = r + (1.0 - luma) * (-0.15) + luma * 0.22
        g_lut = g + (1.0 - luma) * 0.05 + luma * 0.05
        b_lut = b + (1.0 - luma) * 0.25 + luma * (-0.20)
    elif profile == "kodak_film":
        r_lut = np.power(r, 0.90) * 1.05 + 0.02
        g_lut = np.power(g, 0.95) * 1.00 + 0.01
        b_lut = np.power(b, 1.10) * 0.92
    elif profile == "bleach_bypass":
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        overlay = np.where(luma < 0.5, 2 * img_f * luma[:, :, None], 1 - 2 * (1 - img_f) * (1 - luma[:, :, None]))
        desat = 0.65 * overlay + 0.35 * luma[:, :, None]
        r_lut, g_lut, b_lut = desat[:, :, 0], desat[:, :, 1], desat[:, :, 2]
    elif profile == "cross_process":
        r_lut = 0.5 * (1 + np.sin(np.pi * (r - 0.5)))
        g_lut = np.power(g, 0.85)
        b_lut = np.power(b, 1.25) + 0.08 * (1.0 - b)
    else:
        r_lut, g_lut, b_lut = r, g, b

    r_out = (1.0 - strength) * r + strength * r_lut
    g_out = (1.0 - strength) * g + strength * g_lut
    b_out = (1.0 - strength) * b + strength * b_lut
    return np.clip(np.stack([r_out, g_out, b_out], axis=-1) * 255.0, 0, 255).astype(np.uint8)


def vibrance_saturation(image: np.ndarray, vibrance: float = 0.5, saturation: float = 0.0) -> np.ndarray:
    """Selectively boosts muted colors while preserving skin tones."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    s = hsv[:, :, 1]
    norm_s = s / 255.0
    vibrance_boost = (1.0 - norm_s) * float(vibrance) * 255.0
    hsv[:, :, 1] = np.clip(s + vibrance_boost + float(saturation) * s, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def sepia_vintage(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Transforms image into warm vintage antique sepia tones."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ], dtype=np.float32)

    sepia_img = np.clip(cv2.transform(image, sepia_matrix), 0, 255).astype(np.uint8)
    strength = float(np.clip(strength, 0.0, 1.0))
    return cv2.addWeighted(sepia_img, strength, image, 1.0 - strength, 0)


def split_toning(
    image: np.ndarray,
    shadow_hue: float = 210.0,
    shadow_sat: float = 0.35,
    highlight_hue: float = 40.0,
    highlight_sat: float = 0.35
) -> np.ndarray:
    """Tints shadows and highlights with complementary colors."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    img_f = image.astype(np.float32) / 255.0
    luma = 0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 2]
    s_weight = (1.0 - luma)[:, :, None] * float(shadow_sat)
    h_weight = luma[:, :, None] * float(highlight_sat)

    def _hue_to_rgb(h):
        h_rad = np.radians(h % 360.0)
        return np.array([0.5 + 0.5 * np.cos(h_rad), 0.5 + 0.5 * np.cos(h_rad - 2.094), 0.5 + 0.5 * np.cos(h_rad + 2.094)], dtype=np.float32)

    s_rgb = _hue_to_rgb(shadow_hue)
    h_rgb = _hue_to_rgb(highlight_hue)
    graded = img_f + s_weight * (s_rgb - 0.5) + h_weight * (h_rgb - 0.5)
    return np.clip(graded * 255.0, 0, 255).astype(np.uint8)
