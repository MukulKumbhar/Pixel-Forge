"""
ImageLab Pro - Frequency Domain Mechanics & 2D Fourier Analysis
Covers 2D Discrete Fourier Transform (FFT), centered magnitude spectrum,
and frequency-domain filtering: Ideal, Butterworth, and Gaussian (Low-Pass & High-Pass).
Pure NumPy / OpenCV mathematical implementations.
"""

import cv2
import numpy as np


def compute_fft_spectrum(image: np.ndarray) -> np.ndarray:
    """
    Computes 2D centered Fourier magnitude spectrum:
    M(u, v) = c * log(1 + |F(u, v)|)
    Visualizes the spatial frequency distribution (low frequencies centered, high frequencies at periphery).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Compute 2D FFT and shift zero-frequency component to center of spectrum
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)

    # Log magnitude spectrum for dynamic range compression
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1.0)
    # Normalize to [0, 255]
    norm_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    return norm_spectrum.astype(np.uint8)


def frequency_domain_filter(
    image: np.ndarray,
    filter_type: str = "gaussian",
    pass_type: str = "lowpass",
    cutoff_d0: float = 30.0,
    order_n: int = 2
) -> np.ndarray:
    """
    2D Frequency Domain Filtering:
    G(u, v) = H(u, v) * F(u, v)
    - filter_type: 'ideal', 'butterworth', 'gaussian'
    - pass_type: 'lowpass' (smoothing/denoising), 'highpass' (edge extraction)
    - cutoff_d0: Cutoff frequency distance from DC center
    - order_n: Butterworth filter order (sharpness of transition)
    """
    if image.ndim == 3:
        # Process each channel independently for color images
        channels = [frequency_domain_filter(image[:, :, c], filter_type, pass_type, cutoff_d0, order_n) for c in range(3)]
        return np.stack(channels, axis=-1)

    h, w = image.shape[:2]
    # 1. 2D FFT and center shift
    f = np.fft.fft2(image.astype(np.float32))
    fshift = np.fft.fftshift(f)

    # 2. Compute Euclidean distance matrix D(u, v) from center (cx, cy)
    cy, cx = h / 2.0, w / 2.0
    y, x = np.ogrid[:h, :w]
    d_uv = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    eps = 1e-6

    # 3. Construct filter transfer function H(u, v)
    cutoff_d0 = max(1.0, float(cutoff_d0))
    order_n = max(1, int(order_n))

    if filter_type == "ideal":
        # Ideal filter (sharp cutoff, causes ringing artifacts / sinc oscillations)
        h_matrix = np.where(d_uv <= cutoff_d0, 1.0, 0.0)
    elif filter_type == "butterworth":
        # Butterworth filter (smooth transition controlled by order n)
        h_matrix = 1.0 / (1.0 + (d_uv / cutoff_d0) ** (2 * order_n))
    else:  # gaussian
        # Gaussian filter (completely eliminates ringing / smooth Gaussian decay)
        h_matrix = np.exp(-(d_uv ** 2) / (2.0 * (cutoff_d0 ** 2)))

    # Invert for high-pass response
    if pass_type == "highpass":
        h_matrix = 1.0 - h_matrix

    # 4. Multiply in frequency domain
    gshift = fshift * h_matrix

    # 5. Inverse FFT shift & inverse 2D FFT back to spatial domain
    f_ishift = np.fft.ifftshift(gshift)
    img_back = np.fft.ifft2(f_ishift)
    img_real = np.real(img_back)

    return np.clip(img_real, 0, 255).astype(np.uint8)
