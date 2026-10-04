"""
ImageLab Pro - Comprehensive Verification Test Suite
Tests each algorithm in the registry across all 6 domains,
validates pipeline execution, contract handling, metrics, and state management.
"""

import os
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from algorithms.registry import ALGORITHM_REGISTRY, DOMAINS
from core.pipeline import execute_pipeline, detect_state
from core.metrics import calculate_psnr, calculate_mse, calculate_ssim, calculate_entropy, compute_histograms
from core.state_manager import PRESETS, create_layer_dict


def test_all_algorithms():
    print("=== Testing All DIP Algorithms in Registry ===")
    test_rgb = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    test_gray = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    _, test_binary = np.histogram(test_gray, bins=2)
    test_binary = np.where(test_gray > 128, 255, 0).astype(np.uint8)

    passed = 0
    failed = 0

    for algo_key, meta in ALGORITHM_REGISTRY.items():
        func = meta["func"]
        in_c = meta.get("input_contract", "ANY")
        params = {k: v["default"] for k, v in meta.get("params", {}).items()}

        # Pick appropriate test input
        if in_c == "RGB":
            inp = test_rgb.copy()
        elif in_c == "GRAYSCALE":
            inp = test_gray.copy()
        elif in_c == "BINARY":
            inp = test_binary.copy()
        else:
            inp = test_rgb.copy()

        try:
            out = func(inp, **params)
            assert out is not None
            assert isinstance(out, np.ndarray)
            print(f"  [PASS] {algo_key:22s} ({meta['name']}) -> Output shape: {out.shape}, dtype: {out.dtype}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {algo_key:22s} -> Error: {e}")
            failed += 1

    print(f"\nResult: {passed} passed, {failed} failed out of {len(ALGORITHM_REGISTRY)} algorithms.")
    assert failed == 0, f"{failed} algorithms failed execution."


def test_metrics():
    print("\n=== Testing Quantitative Metrics ===")
    img1 = np.ones((100, 100, 3), dtype=np.uint8) * 100
    img2 = np.ones((100, 100, 3), dtype=np.uint8) * 110

    mse = calculate_mse(img1, img2)
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    ent = calculate_entropy(img1)
    hist = compute_histograms(img1)

    print(f"  MSE: {mse:.2f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.3f}")
    print(f"  Entropy: {ent:.2f}")
    assert mse == 100.0
    assert psnr > 0
    assert 0 <= ssim <= 1.0
    print("  [PASS] All metrics calculated successfully.")


def test_presets_in_pipeline():
    print("\n=== Testing Pipeline with Built-in Presets ===")
    base_img = np.random.randint(20, 235, (256, 256, 3), dtype=np.uint8)

    for p_key, p_meta in PRESETS.items():
        print(f"  Testing Preset: {p_meta['title']}")
        layers = []
        for l_spec in p_meta["layers"]:
            layer = create_layer_dict(
                l_spec["algorithm_key"],
                custom_params=l_spec.get("params", {}),
                auto_convert=l_spec.get("auto_convert", True)
            )
            layers.append(layer)

        res = execute_pipeline(base_img, layers)
        print(f"    Total latency: {res.total_time_ms:.2f} ms")
        print(f"    Final shape: {res.output_image.shape}, state: {detect_state(res.output_image)}")
        print(f"    Has conflicts: {res.has_conflicts}")
        assert res.output_image is not None
        assert res.output_image.shape[:2] == base_img.shape[:2]
        print(f"    [PASS] Preset '{p_key}' executed flawlessly.")


if __name__ == "__main__":
    test_all_algorithms()
    test_metrics()
    test_presets_in_pipeline()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
