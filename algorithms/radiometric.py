"""
=============================================================================
ImageLab Pro - Radiometric & Contrast Calibration
Aligned with Academic DIP Lab Experiments (Exp 04, Exp 05, Exp 06)
=============================================================================
Covers:
- Image Negative (Exp 05: s = 255 - r)
- Logarithmic Transformation (Exp 05: s = c * log(1 + r))
- Power-Law / Gamma Transformation (Exp 05: s = c * r^gamma)
- Piecewise Linear Contrast Stretching (Exp 05)
- Bit-Plane Slicing (Exp 05)
- Global Histogram Equalization via CDF (Exp 06)
- Histogram Matching / Specification (Exp 06)
=============================================================================
"""

import cv2
import numpy as np


# ============================================================
# DIP EXPERIMENT REF: EXP 05 - PART 1
# Algorithm: Photographic Negative
# Aim: Invert pixel intensities to enhance bright details in dark regions
# Formula: s = (L - 1) - r = 255 - r
# ============================================================
def image_negative(image: np.ndarray) -> np.ndarray:
    """
    Computes photographic negative by subtracting each pixel value from 255.
    Formula: s = 255 - r
    """
    return 255 - image


# ============================================================
# DIP EXPERIMENT REF: EXP 05 - PART 2
# Algorithm: Logarithmic Transformation
# Aim: Expand dynamic range of dark pixels and compress high bright values
# Formula: s = c * log(1 + r)
# ============================================================
def log_transform(image: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Logarithmic Point Transformation:
    s = c * log(1 + r)
    Standardized so that [0, 255] maps smoothly across [0, 255].
    """
    # Step 1: Calculate normalization constant c so max output equals 255
    c_norm = (255.0 / np.log(1.0 + 255.0)) * float(c)

    # Step 2: Convert to float to avoid overflow during calculation
    img_f = image.astype(np.float32)

    # Step 3: Apply log transfer function: s = c * log(1 + r)
    s = c_norm * np.log(1.0 + img_f)

    # Step 4: Clip to valid 8-bit dynamic range [0, 255]
    return np.clip(s, 0, 255).astype(np.uint8)


# ============================================================
# DIP EXPERIMENT REF: EXP 05 - PART 3
# Algorithm: Power-Law (Gamma) Transformation
# Aim: Correct display non-linearity and adjust midtone contrast
# Formula: s = c * r^gamma
# ============================================================
def power_law_gamma(image: np.ndarray, gamma: float = 1.0, c: float = 1.0) -> np.ndarray:
    """
    Power-Law / Gamma Correction:
    - gamma < 1: Expands shadow/dark midtones (brightens image)
    - gamma > 1: Compresses shadow midtones (darkens image, enhances highlights)
    """
    gamma = max(0.01, float(gamma))
    c = max(0.01, float(c))

    # Step 1: Precompute a 256-entry lookup table (LUT) for O(1) fast mapping
    r = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip(c * (r ** gamma) * 255.0, 0, 255).astype(np.uint8)

    # Step 2: Apply lookup table directly to all pixels
    return cv2.LUT(image, lut)


# ============================================================
# DIP EXPERIMENT REF: EXP 05 - PART 4
# Algorithm: Piecewise Linear Contrast Stretching
# Aim: Stretch a narrow input intensity range [r1, r2] into full range [s1, s2]
# ============================================================
def contrast_stretch_piecewise(
    image: np.ndarray,
    r1: int = 70,
    s1: int = 20,
    r2: int = 180,
    s2: int = 230
) -> np.ndarray:
    """
    Piecewise Linear Contrast Stretching:
    Segment 1: [0, r1] -> [0, s1]
    Segment 2: [r1, r2] -> [s1, s2]
    Segment 3: [r2, 255] -> [s2, 255]
    """
    r1, r2 = min(r1, r2 - 1), max(r2, r1 + 1)
    s1, s2 = np.clip([s1, s2], 0, 255)

    lut = np.zeros(256, dtype=np.uint8)

    # Segment 1: Shadow compression
    slope1 = s1 / max(1, r1)
    lut[:r1] = np.clip(slope1 * np.arange(r1), 0, 255)

    # Segment 2: Midtone contrast expansion (steep slope)
    slope2 = (s2 - s1) / max(1, r2 - r1)
    lut[r1:r2] = np.clip(s1 + slope2 * (np.arange(r1, r2) - r1), 0, 255)

    # Segment 3: Highlight compression
    slope3 = (255 - s2) / max(1, 255 - r2)
    lut[r2:] = np.clip(s2 + slope3 * (np.arange(r2, 256) - r2), 0, 255)

    return cv2.LUT(image, lut)


# ============================================================
# DIP EXPERIMENT REF: EXP 05 - PART 5
# Algorithm: Bit-Plane Slicing
# Aim: Decompose an 8-bit image into 8 binary bit planes (bit 0 to bit 7)
# ============================================================
def bit_plane_slice(image: np.ndarray, plane: int = 7, mode: str = "single") -> np.ndarray:
    """
    Bit-Plane Slicing:
    - plane in [0, 7] (0 = Least Significant Bit, 7 = Most Significant Bit)
    - mode="single": isolates the single binary bit plane scaled to [0, 255]
    - mode="cumulative": reconstructs image keeping bits from selected plane up to 7
    """
    plane = int(np.clip(plane, 0, 7))

    # Convert color image to grayscale to examine bit planes cleanly
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    if mode == "single":
        # Step 1: Shift target bit to LSB and mask with 1, scale {0, 1} to {0, 255}
        sliced = ((gray >> plane) & 1) * 255
        return sliced.astype(np.uint8)

    # Cumulative mode: keep higher significance bits
    mask = 0
    for b in range(plane, 8):
        mask |= (1 << b)
    reconstructed = (gray & mask).astype(np.uint8)
    return reconstructed


# ============================================================
# DIP EXPERIMENT REF: EXP 06 - PART 1
# Algorithm: Global Histogram Equalization
# Aim: Flatten the histogram using the Cumulative Distribution Function (CDF)
# Formula: s_k = round( (L - 1) * CDF(r_k) )
# ============================================================
def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Histogram Equalization:
    Distributes intensity values evenly across the available spectrum [0, 255].
    For RGB color images, equalization is performed on the Y (luminance) channel
    in YCrCb space to preserve natural colors without distortion.
    """
    # Case 1: Grayscale image
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        return cv2.equalizeHist(image.squeeze())

    # Case 2: RGB color image (equalize luminance only)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


# ============================================================
# DIP EXPERIMENT REF: EXP 06 - PART 2
# Algorithm: Histogram Specification / Matching
# Aim: Transform the image histogram to match a desired analytical target shape
# ============================================================
def histogram_specification(image: np.ndarray, target_profile: str = "gaussian") -> np.ndarray:
    """
    Histogram Matching:
    Maps original pixel values through CDF matching so the output
    distribution approximates a target distribution (Gaussian, Bimodal, or Flat).
    """
    def match_channel(src: np.ndarray, target_cdf: np.ndarray) -> np.ndarray:
        # Step 1: Compute source histogram and cumulative distribution function (CDF)
        src_hist, _ = np.histogram(src.flatten(), 256, [0, 256])
        src_cdf = src_hist.cumsum().astype(np.float64)
        src_cdf /= src_cdf[-1]

        # Step 2: Build mapping lookup table by finding closest target CDF values
        lut = np.zeros(256, dtype=np.uint8)
        tgt_idx = 0
        for src_val in range(256):
            while tgt_idx < 255 and target_cdf[tgt_idx] < src_cdf[src_val]:
                tgt_idx += 1
            lut[src_val] = tgt_idx

        # Step 3: Remap pixels using LUT
        return cv2.LUT(src, lut)

    # Step 4: Construct analytical target PDF
    x = np.linspace(0, 255, 256)
    if target_profile == "gaussian":
        pdf = np.exp(-((x - 128) ** 2) / (2 * (40 ** 2)))
    elif target_profile == "bimodal":
        pdf = np.exp(-((x - 64) ** 2) / (2 * (25 ** 2))) + np.exp(-((x - 192) ** 2) / (2 * (25 ** 2)))
    else:
        pdf = np.ones(256)

    target_cdf = pdf.cumsum()
    target_cdf /= target_cdf[-1]

    if image.ndim == 2:
        return match_channel(image, target_cdf)

    # Apply across RGB channels
    matched = np.zeros_like(image)
    for c in range(3):
        matched[:, :, c] = match_channel(image[:, :, c], target_cdf)
    return matched


# ============================================================
# CREATIVE STUDIO ALGORITHM: Linear Brightness & Contrast Adjustment
# Aim: Replicate Photoshop standard Brightness/Contrast sliders
# Formula: g(x, y) = alpha * f(x, y) + beta
# ============================================================
def brightness_contrast(image: np.ndarray, brightness: int = 0, contrast: int = 0) -> np.ndarray:
    """
    Photoshop-style Brightness & Contrast:
    - brightness: [-100, +100] offset
    - contrast: [-100, +100] gain multiplier
    """
    alpha = (contrast + 100.0) / 100.0
    alpha = (alpha ** 1.4) if contrast >= 0 else alpha
    beta = float(brightness)

    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


# ============================================================
# CREATIVE STUDIO ALGORITHM: Solarization (Sabattier Effect)
# Aim: Replicate photographic darkroom partial tone inversion
# Formula: s = 255 - r for pixels > threshold
# ============================================================
def solarize_effect(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    """
    Sabattier Solarization:
    Inverts pixel values that exceed threshold while preserving dark regions.
    """
    img_c = image.copy()
    mask = img_c > threshold
    img_c[mask] = 255 - img_c[mask]
    return img_c


# ============================================================
# CREATIVE STUDIO ALGORITHM: Color Posterization
# Aim: Quantize color channels into discrete graphic poster bands
# ============================================================
def posterize_levels(image: np.ndarray, levels: int = 4) -> np.ndarray:
    """
    Posterize:
    Reduces the number of continuous tonal gradations per channel to discrete steps.
    """
    levels = max(2, min(32, int(levels)))
    step = 255.0 / (levels - 1)
    quantized = np.round(image / (256.0 / levels)) * step
    return np.clip(quantized, 0, 255).astype(np.uint8)

