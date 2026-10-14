"""
Frequency domain filtering and Fourier spectrum analysis.
"""

import cv2
import numpy as np


def compute_fft_spectrum(image: np.ndarray) -> np.ndarray:
    """Compute centered 2D Fourier magnitude spectrum."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1.0)
    return cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def frequency_domain_filter(
    image: np.ndarray,
    filter_type: str = "gaussian",
    pass_type: str = "lowpass",
    cutoff_d0: float = 30.0,
    order_n: int = 2
) -> np.ndarray:
    """Apply ideal, Butterworth, or Gaussian frequency domain filtering."""
    if image.ndim == 3:
        channels = [frequency_domain_filter(image[:, :, c], filter_type, pass_type, cutoff_d0, order_n) for c in range(3)]
        return np.stack(channels, axis=-1)

    h, w = image.shape[:2]
    f = np.fft.fft2(image.astype(np.float32))
    fshift = np.fft.fftshift(f)

    cy, cx = h / 2.0, w / 2.0
    y, x = np.ogrid[:h, :w]
    d_uv = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    cutoff_d0 = max(1.0, float(cutoff_d0))
    order_n = max(1, int(order_n))

    if filter_type == "ideal":
        h_matrix = np.where(d_uv <= cutoff_d0, 1.0, 0.0)
    elif filter_type == "butterworth":
        h_matrix = 1.0 / (1.0 + (d_uv / cutoff_d0) ** (2 * order_n))
    else:
        h_matrix = np.exp(-(d_uv ** 2) / (2.0 * (cutoff_d0 ** 2)))

    if pass_type == "highpass":
        h_matrix = 1.0 - h_matrix

    gshift = fshift * h_matrix
    f_ishift = np.fft.ifftshift(gshift)
    img_back = np.fft.ifft2(f_ishift)
    return np.clip(np.real(img_back), 0, 255).astype(np.uint8)
