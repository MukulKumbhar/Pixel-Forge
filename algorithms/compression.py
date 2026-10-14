"""
=============================================================================
Pixel-Forge - Information Redundancy & Image Compression Engine
Aligned with Academic DIP (Gonzalez & Woods Ch. 8: Image Compression)
=============================================================================
Covers:
- Academic Discrete Cosine Transform (DCT) 8x8 Block JPEG Simulation
- Color Quantization & Palette Compression (Median-Cut / NeuQuant)
- Production Lossless Compression (WebP VP8L, Optimized Deflate PNG)
- Production Deep Lossy Compression (WebP VP8, JPEG DCT, Downscaling)
- Quantitative Fidelity Telemetry (PSNR, SSIM, MSE, Compression Ratio, % Saved)
=============================================================================
"""

import io
import time
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np
from PIL import Image



# =============================================================================
# STANDARD JPEG LUMINANCE QUANTIZATION MATRIX (ISO/IEC 10918-1)
# Derived from human psycho-visual contrast sensitivity function (CSF)
# =============================================================================
JPEG_LUM_QTABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)


# =============================================================================
# ALGORITHM: Academic 8x8 Block DCT JPEG Compression Simulation
# Aim: Demonstrates textbook DIP JPEG baseline encoding:
#      1. Partition into 8x8 non-overlapping spatial blocks
#      2. Forward 2D-DCT energy compaction
#      3. Psychovisual quantization scaling via user Quality factor (1-100)
#      4. Inverse quantization and 2D-IDCT reconstruction
# =============================================================================
def dct_block_jpeg_simulation(
    image: np.ndarray,
    quality: int = 50,
    block_size: int = 8
) -> np.ndarray:
    """
    Simulates textbook JPEG 8x8 block-based DCT compression and frequency quantization.
    Demonstrates blocking artifacts and high-frequency ringing at low quality settings.
    """
    quality = int(np.clip(quality, 1, 100))
    h, w = image.shape[:2]

    # Step 1: Compute standard JPEG quality scaling factor
    if quality < 50:
        scale = 50.0 / quality
    else:
        scale = (100.0 - quality) / 50.0

    q_table = np.clip(np.round(JPEG_LUM_QTABLE * scale), 1.0, 255.0)

    # Process either multi-channel RGB or single-channel Grayscale
    is_color = (image.ndim == 3 and image.shape[2] >= 3)
    channels = [image[:, :, c] for c in range(3)] if is_color else [image.copy()]
    reconstructed_channels = []

    # Pad image dimensions to multiples of block_size (8)
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size

    for ch in channels:
        if pad_h > 0 or pad_w > 0:
            padded = cv2.copyMakeBorder(ch, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        else:
            padded = ch.copy()

        ph, pw = padded.shape[:2]
        output_plane = np.zeros((ph, pw), dtype=np.float32)

        # Iterate over 8x8 non-overlapping blocks
        for r in range(0, ph, block_size):
            for c in range(0, pw, block_size):
                # Level shift: [0, 255] -> [-128, 127]
                block = padded[r:r+block_size, c:c+block_size].astype(np.float32) - 128.0

                # 2D Forward Discrete Cosine Transform
                dct_block = cv2.dct(block)

                # Quantization: discard psychovisually redundant high-frequency coefficients
                quantized = np.round(dct_block / q_table)

                # De-quantization (reconstruction)
                dequantized = quantized * q_table

                # 2D Inverse Discrete Cosine Transform
                idct_block = cv2.idct(dequantized) + 128.0
                output_plane[r:r+block_size, c:c+block_size] = idct_block

        # Crop back to original dimensions and clamp to [0, 255]
        unpadded = output_plane[:h, :w]
        reconstructed_channels.append(np.clip(unpadded, 0, 255).astype(np.uint8))

    if is_color:
        return np.stack(reconstructed_channels, axis=2)
    return reconstructed_channels[0]


# =============================================================================
# ALGORITHM: Paletted Color Quantization (PNG-8 / GIF Palette Compression)
# Aim: Demonstrates color palette reduction from 24-bit (16.7M colors)
#      to 1-8 bit (2 to 256 colors) via Median Cut and Floyd-Steinberg dithering.
# =============================================================================
def paletted_color_quantization(
    image: np.ndarray,
    num_colors: int = 32,
    dither: bool = True
) -> np.ndarray:
    """
    Reduces color palette to specified number of colors (2 to 256).
    Simulates GIF/PNG-8 indexed palette compression.
    """
    num_colors = int(np.clip(num_colors, 2, 256))

    # Convert numpy array to PIL Image
    if image.ndim == 3 and image.shape[2] == 4:
        pil_img = Image.fromarray(image, mode="RGBA").convert("RGB")
    elif image.ndim == 3:
        pil_img = Image.fromarray(image, mode="RGB")
    else:
        pil_img = Image.fromarray(image).convert("RGB")

    dither_flag = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    quantized_pil = pil_img.quantize(colors=num_colors, method=Image.Quantize.MEDIANCUT, dither=dither_flag)
    result_rgb = quantized_pil.convert("RGB")

    return np.array(result_rgb, dtype=np.uint8)


# =============================================================================
# PRODUCTION COMPRESSOR ENGINE: Lossless & Deep Lossy Dual Pipeline
# =============================================================================
def compress_image_stream(
    image: np.ndarray,
    mode: str = "lossless",
    format_choice: str = "webp",
    quality: int = 80,
    effort: int = 6,
    scale_factor: float = 1.0,
    chroma_subsampling: str = "4:2:0",
    strip_metadata: bool = True,
    num_palette_colors: int = 64
) -> Dict[str, Any]:
    """
    Industrial-grade, high-performance image compression engine supporting:
    1. 'lossless': 100% bit-exact or lossless encoding with zero quality degradation (WebP Lossless, PNG Deflate).
    2. 'lossy': Deep compression with interactive quality slider (WebP, JPEG, Quantized PNG),
                optional dimension downscaling, and comprehensive fidelity telemetry.

    Returns:
        dict containing:
        - 'compressed_bytes': bytes
        - 'decompressed_image': np.ndarray (for live visual inspection)
        - 'orig_size_bytes': int
        - 'compressed_size_bytes': int
        - 'compression_ratio': float (e.g. 8.4 : 1)
        - 'space_saved_percent': float (e.g. 88.1%)
        - 'psnr': float (dB)
        - 'ssim': float (0.0 to 1.0)
        - 'mse': float
        - 'mime_type': str
        - 'file_extension': str
        - 'elapsed_ms': float
    """
    t_start = time.perf_counter()

    # Determine input dimensions & format
    h, w = image.shape[:2]
    has_alpha = (image.ndim == 3 and image.shape[2] == 4)

    # Calculate uncompressed baseline size (raw uncompressed pixels)
    channels_count = image.shape[2] if image.ndim == 3 else 1
    raw_uncompressed_bytes = h * w * channels_count

    # Prepare PIL image
    if has_alpha:
        pil_src = Image.fromarray(image, mode="RGBA")
    elif image.ndim == 3:
        pil_src = Image.fromarray(image, mode="RGB")
    else:
        pil_src = Image.fromarray(image).convert("RGB")

    # Optional deep compression downscaling
    scale_factor = float(np.clip(scale_factor, 0.1, 1.0))
    if scale_factor < 0.999:
        target_w = max(16, int(w * scale_factor))
        target_h = max(16, int(h * scale_factor))
        working_pil = pil_src.resize((target_w, target_h), Image.Resampling.LANCZOS)
    else:
        working_pil = pil_src.copy()

    format_choice = format_choice.lower().strip()
    buffer = io.BytesIO()

    # -------------------------------------------------------------
    # CASE 1: LOSSLESS COMPRESSION (Zero Quality Loss Guaranteed)
    # -------------------------------------------------------------
    if mode.lower() == "lossless":
        if format_choice == "webp":
            # WebP VP8L Lossless encoding (RFC 7914)
            # method 0..6 controls compression effort vs CPU time
            effort_clamped = int(np.clip(effort, 0, 6))
            working_pil.save(
                buffer,
                format="WEBP",
                lossless=True,
                quality=100,
                method=effort_clamped
            )
            mime_type = "image/webp"
            file_extension = "webp"

        else:
            # Default to PNG Optimized Deflate
            # compress_level 1..9 controls zlib compression ratio
            effort_clamped = int(np.clip(effort, 1, 9))
            working_pil.save(
                buffer,
                format="PNG",
                optimize=True,
                compress_level=effort_clamped
            )
            mime_type = "image/png"
            file_extension = "png"

    # -------------------------------------------------------------
    # CASE 2: DEEP LOSSY COMPRESSION (Quality Slider Controlled)
    # -------------------------------------------------------------
    else:
        quality = int(np.clip(quality, 1, 100))

        if format_choice == "webp":
            # WebP Lossy VP8
            effort_clamped = int(np.clip(effort, 0, 6))
            working_pil.save(
                buffer,
                format="WEBP",
                lossless=False,
                quality=quality,
                method=effort_clamped
            )
            mime_type = "image/webp"
            file_extension = "webp"

        elif format_choice in ["png_quantized", "png_palette"]:
            # Paletted Color Quantization PNG
            num_colors = int(np.clip(num_palette_colors, 2, 256))
            quantized = working_pil.convert("RGB").quantize(
                colors=num_colors,
                method=Image.Quantize.MEDIANCUT,
                dither=Image.Dither.FLOYDSTEINBERG
            )
            quantized.save(
                buffer,
                format="PNG",
                optimize=True,
                compress_level=6
            )
            mime_type = "image/png"
            file_extension = "png"

        else:
            # JPEG DCT Lossy Compression
            # JPEG doesn't support alpha channel - blend over white or drop
            if working_pil.mode in ("RGBA", "LA"):
                bg = Image.new("RGB", working_pil.size, (255, 255, 255))
                bg.paste(working_pil, mask=working_pil.split()[-1])
                jpeg_pil = bg
            else:
                jpeg_pil = working_pil.convert("RGB")

            # Map chroma subsampling
            subsampling_flag = 0 if chroma_subsampling == "4:4:4" else 2  # 0=4:4:4, 2=4:2:0
            jpeg_pil.save(
                buffer,
                format="JPEG",
                quality=quality,
                optimize=True,
                subsampling=subsampling_flag
            )
            mime_type = "image/jpeg"
            file_extension = "jpg"

    compressed_bytes = buffer.getvalue()
    compressed_size_bytes = len(compressed_bytes)

    # Decompress back to numpy array for fidelity evaluation
    buffer.seek(0)
    reconstructed_pil = Image.open(buffer)
    reconstructed_np = np.array(reconstructed_pil)

    # Compute fidelity metrics against original image
    # Note: If dimensions were scaled down, upsample back for MSE/PSNR/SSIM calculation
    if reconstructed_np.shape[:2] != (h, w):
        eval_proc = cv2.resize(reconstructed_np, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        eval_proc = reconstructed_np.copy()

    # Calculate metrics
    eval_orig = image.copy()
    if eval_orig.ndim == 3 and eval_orig.shape[2] == 4:
        eval_orig = cv2.cvtColor(eval_orig, cv2.COLOR_RGBA2RGB)
    if eval_proc.ndim == 3 and eval_proc.shape[2] == 4:
        eval_proc = cv2.cvtColor(eval_proc, cv2.COLOR_RGBA2RGB)

    from core.metrics import calculate_mse, calculate_psnr, calculate_ssim
    mse_val = calculate_mse(eval_orig, eval_proc)
    psnr_val = calculate_psnr(eval_orig, eval_proc)
    ssim_val = calculate_ssim(eval_orig, eval_proc)

    # In lossless mode with full resolution, guarantee perfect zero loss
    if mode.lower() == "lossless" and scale_factor >= 0.999:
        if mse_val < 1e-4:
            mse_val = 0.0
            psnr_val = 99.9
            ssim_val = 1.0

    # Compression calculations
    # Calculate uncompressed PNG / Raw baseline size for fair comparison
    compression_ratio = float(raw_uncompressed_bytes) / max(1, compressed_size_bytes)
    space_saved_percent = max(0.0, (1.0 - (float(compressed_size_bytes) / raw_uncompressed_bytes)) * 100.0)

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    return {
        "compressed_bytes": compressed_bytes,
        "decompressed_image": reconstructed_np,
        "orig_size_bytes": raw_uncompressed_bytes,
        "compressed_size_bytes": compressed_size_bytes,
        "compression_ratio": compression_ratio,
        "space_saved_percent": space_saved_percent,
        "psnr": psnr_val,
        "ssim": ssim_val,
        "mse": mse_val,
        "mime_type": mime_type,
        "file_extension": file_extension,
        "elapsed_ms": elapsed_ms
    }
