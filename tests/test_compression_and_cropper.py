"""
=============================================================================
Pixel-Forge - Test Suite for Image Compressor & Vector Shape Cropper
Validates:
1. Lossless compression bit-exact zero-loss guarantee (WebP Lossless, PNG Deflate)
2. Lossy compression quality slider scaling and deep compression
3. Vector shape cropping across all 11 geometric shapes (Circle, Square, Star, Heart, etc.)
4. Transparency (RGBA) handling, feathering, and tight canvas bounding box trimming
5. Non-destructive pipeline integration with shape crop layers
=============================================================================
"""

import os
import sys
import numpy as np
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms.compression import compress_image_stream, dct_block_jpeg_simulation, paletted_color_quantization
from algorithms.geometry import vector_shape_crop
from core.pipeline import execute_pipeline, detect_state, convert_state
from core.state_manager import create_layer_dict


@pytest.fixture
def sample_rgb_image():
    """Generates a synthetic gradient test image."""
    h, w = 256, 256
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    r = xx
    g = yy
    b = ((xx.astype(int) + yy.astype(int)) // 2).astype(np.uint8)
    return np.dstack([r, g, b])


def test_lossless_webp_compression(sample_rgb_image):
    """Verifies that Lossless WebP achieves byte reduction with 0.0 MSE."""
    res = compress_image_stream(
        image=sample_rgb_image,
        mode="lossless",
        format_choice="webp",
        effort=6
    )

    assert res["compressed_bytes"] is not None
    assert len(res["compressed_bytes"]) > 0
    assert res["compressed_size_bytes"] < res["orig_size_bytes"]
    assert res["space_saved_percent"] > 0.0
    assert res["mse"] == 0.0
    assert res["psnr"] >= 99.0
    assert res["ssim"] == 1.0
    assert res["mime_type"] == "image/webp"
    assert res["file_extension"] == "webp"
    assert res["decompressed_image"].shape == sample_rgb_image.shape


def test_lossless_png_compression(sample_rgb_image):
    """Verifies that Lossless PNG achieves bit-exact match with 0.0 MSE."""
    res = compress_image_stream(
        image=sample_rgb_image,
        mode="lossless",
        format_choice="png",
        effort=9
    )

    assert res["compressed_size_bytes"] < res["orig_size_bytes"]
    assert res["mse"] == 0.0
    assert res["psnr"] >= 99.0
    assert res["ssim"] == 1.0
    assert res["file_extension"] == "png"


def test_lossy_quality_slider_scaling(sample_rgb_image):
    """Verifies that decreasing quality slider reduces file size monotonically."""
    res_high = compress_image_stream(
        image=sample_rgb_image,
        mode="lossy",
        format_choice="webp",
        quality=90
    )

    res_deep = compress_image_stream(
        image=sample_rgb_image,
        mode="lossy",
        format_choice="webp",
        quality=20
    )

    # Deep compression should yield smaller file size and higher compression ratio
    assert res_deep["compressed_size_bytes"] < res_high["compressed_size_bytes"]
    assert res_deep["compression_ratio"] > res_high["compression_ratio"]
    assert res_deep["space_saved_percent"] > res_high["space_saved_percent"]


def test_deep_compression_downscaling(sample_rgb_image):
    """Verifies that deep compression dimension downscaling works."""
    res = compress_image_stream(
        image=sample_rgb_image,
        mode="lossy",
        format_choice="jpeg",
        quality=30,
        scale_factor=0.5
    )

    assert res["compressed_size_bytes"] < 5000  # Extremely small
    assert res["compression_ratio"] > 10.0


def test_shape_cropper_all_shapes(sample_rgb_image):
    """Verifies that all 11 geometric shapes produce valid cropped outputs."""
    shapes = [
        "circle", "square", "rectangle", "ellipse", "triangle",
        "diamond", "hexagon", "octagon", "star", "heart", "polygon"
    ]

    for shape in shapes:
        out = vector_shape_crop(
            image=sample_rgb_image,
            shape=shape,
            center_x=0.5,
            center_y=0.5,
            size=0.75,
            rotation=15.0,
            feather=2,
            bg_fill="transparent",
            tight_crop=True
        )

        assert out is not None
        assert isinstance(out, np.ndarray)
        assert out.ndim == 3
        # Transparent mode must yield RGBA (4 channels)
        assert out.shape[2] == 4
        # Verify alpha channel has non-zero pixels inside shape
        alpha = out[:, :, 3]
        assert np.count_nonzero(alpha > 0) > 0


def test_shape_cropper_solid_backgrounds(sample_rgb_image):
    """Verifies that solid background fill modes work correctly."""
    for bg_mode in ["black", "white", "studio_dark"]:
        out = vector_shape_crop(
            image=sample_rgb_image,
            shape="circle",
            bg_fill=bg_mode,
            tight_crop=False
        )
        assert out.ndim == 3
        assert out.shape[2] == 3  # RGB with solid background
        assert out.shape[:2] == sample_rgb_image.shape[:2]


def test_pipeline_shape_crop_integration(sample_rgb_image):
    """Verifies that a shape crop layer can be placed in the pipeline."""
    crop_layer = create_layer_dict(
        "geo_shape_crop",
        custom_params={
            "shape": "heart",
            "size": 0.8,
            "bg_fill": "transparent",
            "tight_crop": True
        }
    )

    res = execute_pipeline(sample_rgb_image, [crop_layer])
    assert res.output_image is not None
    assert detect_state(res.output_image) == "RGBA"
    assert res.output_image.shape[2] == 4


def test_academic_dct_jpeg_simulation(sample_rgb_image):
    """Verifies academic 8x8 block DCT JPEG simulation."""
    reconstructed = dct_block_jpeg_simulation(sample_rgb_image, quality=25)
    assert reconstructed.shape == sample_rgb_image.shape
    assert reconstructed.dtype == np.uint8


def test_academic_paletted_quantization(sample_rgb_image):
    """Verifies color palette quantization."""
    quantized = paletted_color_quantization(sample_rgb_image, num_colors=16)
    assert quantized.shape == sample_rgb_image.shape
    assert quantized.dtype == np.uint8


def test_large_image_handling():
    """Verifies that large 25MB+ / 3000x3000px image arrays process without failure."""
    import cv2
    large_img = np.zeros((3000, 3000, 3), dtype=np.uint8)
    large_img[:1500, :] = 200
    large_img[1500:, :] = 80

    # Test proxy downsampling
    scale = 2048.0 / max(large_img.shape[:2])
    proxy = cv2.resize(large_img, (int(3000 * scale), int(3000 * scale)), interpolation=cv2.INTER_AREA)
    assert max(proxy.shape[:2]) == 2048

    # Test lossless compression on large image
    comp_res = compress_image_stream(large_img, mode="lossless", format_choice="webp", effort=4)
    assert comp_res["compressed_bytes"] is not None
    assert comp_res["space_saved_percent"] > 0
    assert comp_res["mse"] == 0.0


def test_crop_reset_preserves_other_pipeline_layers(sample_rgb_image):
    """
    Verifies that removing / resetting crop layers completely restores the
    full original uncropped scale WITHOUT disturbing, deleting, or altering
    any other active filters in the pipeline stack.
    """
    import streamlit as st
    from core.state_manager import init_session_state, has_crop_layer, remove_crop_layers_from_stack
    from algorithms.registry import ALGORITHM_REGISTRY

    init_session_state()

    # Build multi-layer stack: Brightness -> Shape Crop -> Bilateral Blur -> Sepia
    layer1 = create_layer_dict("rad_brightness_contrast", {"brightness": 25, "contrast": 30})
    layer2 = create_layer_dict("geo_shape_crop", {"shape": "circle", "size": 0.5, "tight_crop": True})
    layer3 = create_layer_dict("rest_bilateral", {"d": 5, "sigma_color": 50.0})
    layer4 = create_layer_dict("col_sepia", {"strength": 0.8})

    st.session_state.active_layer_stack = [layer1, layer2, layer3, layer4]
    st.session_state.selected_layer_id = layer2["id"]

    assert has_crop_layer() is True
    assert len(st.session_state.active_layer_stack) == 4

    # Run pipeline with crop active
    res_cropped = execute_pipeline(sample_rgb_image, st.session_state.active_layer_stack)
    # Output should be cropped to smaller bounding box
    assert res_cropped.output_image.shape[0] < sample_rgb_image.shape[0]

    # Now execute Crop Reset
    remove_crop_layers_from_stack()

    # Verify crop layer is gone but all other 3 filters remain intact in order
    assert has_crop_layer() is False
    remaining_stack = st.session_state.active_layer_stack
    assert len(remaining_stack) == 3

    assert remaining_stack[0]["algorithm_key"] == "rad_brightness_contrast"
    assert remaining_stack[0]["params"]["brightness"] == 25
    assert remaining_stack[0]["params"]["contrast"] == 30

    assert remaining_stack[1]["algorithm_key"] == "rest_bilateral"
    assert remaining_stack[1]["params"]["d"] == 5

    assert remaining_stack[2]["algorithm_key"] == "col_sepia"
    assert remaining_stack[2]["params"]["strength"] == 0.8

    # Re-run pipeline: Output image must have full original dimensions restored
    res_restored = execute_pipeline(sample_rgb_image, remaining_stack)
    assert res_restored.output_image.shape[:2] == sample_rgb_image.shape[:2]
    # And the filters were still applied (image is modified compared to raw input)
    assert not np.array_equal(res_restored.output_image, sample_rgb_image)


def test_all_algorithms_have_friendly_names_and_descriptions():
    """
    Verifies that all 69 DIP algorithms in ALGORITHM_REGISTRY have consumer-friendly
    names, plain-language descriptions, and hover tooltips for global users.
    """
    from algorithms.registry import ALGORITHM_REGISTRY, DOMAINS

    assert len(ALGORITHM_REGISTRY) >= 69

    for key, meta in ALGORITHM_REGISTRY.items():
        assert "friendly_name" in meta, f"Missing friendly_name for {key}"
        assert len(meta["friendly_name"].strip()) > 0, f"Empty friendly_name for {key}"

        assert "simple_description" in meta, f"Missing simple_description for {key}"
        assert len(meta["simple_description"].strip()) > 0, f"Empty simple_description for {key}"

        assert "tooltip" in meta, f"Missing tooltip for {key}"
        assert len(meta["tooltip"].strip()) > 0, f"Empty tooltip for {key}"

        assert "name" in meta, f"Missing academic name for {key}"
        assert "domain" in meta, f"Missing domain for {key}"
        assert meta["domain"] in DOMAINS, f"Invalid domain {meta['domain']} for {key}"


