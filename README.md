# 🔬 Pixel-Forge — Interactive Digital Image Processing Studio

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV%20%7C%20NumPy-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/Tests-66%2F66%20Passing-brightgreen.svg)](tests/test_pipeline.py)

**Pixel-Forge** is an industry-grade, non-destructive Digital Image Processing (DIP) studio and creative digital content creation (DCC) workbench. It brings together the complete academic DIP curriculum (**Experiments 01 to 10**) alongside a full-featured **Adobe Photoshop & DaVinci Resolve creative suite**—featuring 66 real-time algorithms across 7 dedicated domains.

---

## 📑 Table of Contents

1. [Quickstart in 3 Steps](#-quickstart-in-3-steps)
2. [Dual Scope: Academic Foundation + Photoshop Creative Studio](#-dual-scope-academic-foundation--photoshop-creative-studio)
3. [Academic Lab Experiment Mapping (Exp 01 — 10)](#-academic-lab-experiment-mapping-exp-01--10)
4. [Creative Studio & Photoshop Replica Suite](#-creative-studio--photoshop-replica-suite)
5. [Studio Architecture & Layout](#-studio-architecture--layout)
6. [Key Features & Usability Upgrades](#-key-features--usability-upgrades)
7. [Industry & Creative Workflow Presets](#-industry--creative-workflow-presets)
8. [Detailed Sub-Documentation](#-detailed-sub-documentation)
9. [Verification & Automated Test Suite](#-verification--automated-test-suite)

---

## 🚀 Quickstart in 3 Steps

### 1. Clone & Enter Directory
```bash
git clone https://github.com/MukulKumbhar/Pixel-Forge.git
cd Pixel-Forge
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Studio
```bash
streamlit run app.py
```
> The studio will automatically open in your default browser at `http://localhost:8501`.

---

## 🌟 Dual Scope: Academic Foundation + Photoshop Creative Studio

Pixel-Forge was engineered to provide **both**:
1. **Academic Rigor:** 100% adherence to standard university DIP syllabi and laboratory experiments (discretization, sampling, point operations, CDF histogram equalization, spatial filtering, trigonometric HSI colorimetry, differential edge detectors, Hough transforms, morphological mechanics, and 2D Fourier FFT).
2. **Professional Creative Power:** A full Photoshop / DaVinci Resolve replica toolset (Unsharp Masking, Bilateral skin retouching, Pencil Sketch, Cel-Shaded Cartoonify, Bas-Relief Emboss, Radial Vignette, Tilt-Shift Miniature, Fisheye Lens Distortion, Keystone Perspective Warp, Smart Vibrance, Antique Sepia, Split-Toning, Adaptive Thresholding, Hough Circles, and Harris Corners).

---

## 🎓 Academic Lab Experiment Mapping (Exp 01 — 10)

| Lab Experiment | Experiment Title | Implemented Algorithms | Pixel-Forge Module |
| :---: | :--- | :--- | :--- |
| **Exp 01** | Image I/O & Channel Properties | Resolution extraction, channel inspection, synthetic generator | [`geometry.py`](file:///d:/Academics/Digital_Image_Processing/Project/ImageLab/algorithms/geometry.py) |
| **Exp 02** | Pixel Access & RGB Channels | R, G, B channel splitting, luminance extraction, color scopes | [`colorimetry.py`](file:///d:/Academics/Digital_Image_Processing/Project/ImageLab/algorithms/colorimetry.py) |
| **Exp 03** | Sampling & Quantization | Spatial downsampling, Nearest/Bilinear/Bicubic, Bit-depth (8 to 1 bit) | [`geometry.py`](file:///d:/Academics/Digital_Image_Processing/Project/ImageLab/algorithms/geometry.py) |
| **Exp 04** | Arithmetic Operations | Multi-frame addition, subtraction, alpha blending, split difference | [`core/metrics.py`](file:///d:/Academics/Digital_Image_Processing/Project/ImageLab/core/metrics.py) |
| **Exp 05** | Point Transformations | Negative ($s = 255-r$), Log ($s = c\log(1+r)$), Gamma ($s = cr^\gamma$), Piecewise Stretch, Bit-Planes | [`radiometric.py`](file:///d:/Academics/Digital_Image_Processing/Project/ImageLab/algorithms/radiometric.py) |
| **Exp 06** | Histogram Processing | Global Histogram Equalization (CDF), Histogram Matching / Specification | [`radiometric.py`](file:///d:/Academics/Digital_Image_Processing/Project/ImageLab/algorithms/radiometric.py) |
| **Exp 07** | Spatial Domain Filtering | Linear Averaging (Mean) Filter, Non-Linear Median Filter, Min/Max/Midpoint | [`restoration.py`](file:///d:/Academics/Digital_Image_Processing/Project/ImageLab/algorithms/restoration.py) |
| **Exp 08** | Color Models & Colorimetry | Trigonometric RGB ↔ HSI Conversion, HSI Vector Grading, CMYK, White Balance | [`colorimetry.py`](file:///d:/Academics/Digital_Image_Processing/Project/ImageLab/algorithms/colorimetry.py) |
| **Exp 09** | Edge Detection | Sobel ($G_x, G_y$), Prewitt, Laplacian second-derivative, Canny Hysteresis | [`segmentation.py`](file:///d:/Academics/Digital_Image_Processing/Project/ImageLab/algorithms/segmentation.py) |
| **Exp 10** | Hough Transform | Hough Line Transform ($\rho = x\cos\theta + y\sin\theta$) vector extraction | [`segmentation.py`](file:///d:/Academics/Digital_Image_Processing/Project/ImageLab/algorithms/segmentation.py) |

---

## 🎨 Creative Studio & Photoshop Replica Suite

Beyond standard syllabus experiments, Pixel-Forge equips you with professional DCC capabilities:

1. **Photoshop Unsharp Masking (USM):** Crisp definition enhancement with adjustable amount, radius, and noise-floor threshold.
2. **Bilateral Filter (Smart Blur / Skin Retouch):** Non-linear edge-preserving filter that smooths facial textures and blemishes while locking eyes and lips sharp.
3. **Hand-Drawn Pencil Sketch:** Color Dodge blending of inverted Gaussian blur with grayscale raster for genuine graphite art.
4. **Cel-Shaded Cartoonify:** Multi-pass bilateral color quantization combined with adaptive Laplacian ink outline overlays.
5. **3D Bas-Relief Directional Emboss:** Directional lighting convolution simulating stamped metal and carved stone relief.
6. **Radial Optical Vignette:** Quadratic falloff spotlight shading simulating vintage Leica 35mm street photography lenses.
7. **Tilt-Shift Miniature Photography:** Smooth defocus gradient outside a selectable sharp focal band turning real scenes into toy dioramas.
8. **Optical Lens Distortion:** Radial barrel and pincushion polynomial distortion simulating GoPro fisheye cameras or correcting architecture.
9. **Perspective Keystone Quad Transform:** Four-corner planar homography for straightening angled document scans and tilted buildings.
10. **Smart Vibrance:** Non-linear saturation boosting that protects skin tones from oversaturation.
11. **Antique Sepia Tone:** Professional 3x3 matrix photographic silver print toning.
12. **DaVinci Resolve Split Toning:** Complementary shadow and highlight chromatic split grading.
13. **Local Adaptive Binarization:** Local window thresholding for documents under casting shadows or non-uniform desk lamps.
14. **Hough Circle Transform:** Automated gradient voting for circular objects, pupils, and coins.
15. **Harris Corner Detection:** Second-moment auto-correlation matrix ($R = \det(M) - k \cdot \text{trace}(M)^2$) for keypoint extraction.

---

## 🏛️ Studio Architecture & Layout

```
Pixel-Forge/
├── app.py                     # Studio entry point: Orchestrates layout, state, and rendering
├── core/
│   ├── pipeline.py            # Non-destructive layer execution manager & state contract validator
│   ├── state_manager.py       # Session-state serialization, undo/redo history & 9 workflow presets
│   └── metrics.py             # Quantitative diagnostics: PSNR, MSE, SSIM, Entropy, Histograms
├── algorithms/                # Pure mathematical functions (Zero UI dependencies)
│   ├── README.md              # 📖 Comprehensive mathematical reference & algorithm directory
│   ├── geometry.py            # Sampling, quantization, 2D affine, canvas resize, lens distortion, keystone
│   ├── radiometric.py         # Negative, log, gamma, contrast stretch, bit-planes, hist eq, solarize, posterize
│   ├── restoration.py         # Noise generators, mean, median, Gaussian blur, motion blur, bilateral, USM
│   ├── colorimetry.py         # HSI vector grading, CMYK, lift/gamma/gain, white balance, vibrance, sepia, split-tone
│   ├── segmentation.py        # Sobel, Prewitt, Laplacian, Canny, Otsu, Hough lines/circles, adaptive thresh, Harris
│   ├── morphology.py          # Structuring elements, dilation, erosion, open, close, gradient, top-hat, black-hat
│   ├── frequency.py           # 2D Fourier FFT magnitude spectrum, Butterworth & Gaussian filters
│   ├── artistic.py            # Pencil sketch, cartoonify, bas-relief emboss, radial vignette, tilt-shift
│   └── registry.py            # Central registry: 66 algorithm definitions across 7 domains
├── ui/
│   ├── README.md              # 📖 UI layout guide: Sticky viewport, scrollable shelf & CSS theme
│   ├── styles.py              # Studio dark-mode DCC design system & anti-flicker CSS overrides
│   ├── viewport.py            # Fixed sticky dual-viewport canvas & real-time telemetry HUD
│   ├── parameter_drawer.py    # Synchronized sliders, hot-swapping & filter micro-info cards
│   ├── bottom_tray.py         # Independently scrollable bottom tool dock (7 Domains)
│   └── conflict_resolver.py   # State contract validator & auto-convert shims (RGB ↔ Gray ↔ Binary)
├── assets/
│   ├── generate_samples.py    # Synthetic standard reference image generator
│   └── sample_images/         # Standard benchmark images (Chroma, Siemens Star, OCR, Shapes)
├── tests/
│   └── test_pipeline.py       # Automated verification test suite (66 algorithms & 9 presets)
└── requirements.txt           # Project dependencies
```

---

## 🌟 Key Features & Usability Upgrades

### 1. 📌 Pinned Sticky Dual-Viewport Canvas
- The image comparison canvas remains **pinned at the top** (`position: sticky; top: 2.875rem;`).
- You can adjust sliders or pick filters without ever scrolling away from the image preview!
- Four interactive viewport comparison modes:
  - **Side-by-Side:** Direct side-by-side comparison of source frame and processed output.
  - **Interactive Split Wipe:** Dynamic horizontal slider revealing the before/after divider.
  - **Difference Heatmap:** Color-mapped residual error magnitude ($|I_{\text{source}} - I_{\text{processed}}|$) showing exact pixel changes.
  - **Output Only:** Clean full-canvas view of the final processed output.

### 2. 🗂️ Independently Scrollable Bottom Filter Shelf
- The bottom filter catalog is placed inside a dedicated container with independent scrolling (`overflow-y: auto`).
- Just like the sidebar, scrolling over the bottom shelf scrolls *only the filters*, leaving the page and image preview anchored in place.

### 3. ⚡ Zero-Loop Preset Manager & Anti-Flicker CSS
- Eliminated recursive rerun loops when applying presets using a state-resetting `on_change` callback.
- Added **1-Click Quick Preset Buttons** for instant, single-click workflow loading.
- Suppressed Streamlit's jarring page fade-out opacity transition for smooth, instantaneous updates.

### 4. 💡 Filter Micro-Info Cards
- Compact, student-friendly info badge for every tool:
  - 💡 **What it does:** 1-sentence plain-English definition.
  - 🎯 **Example:** Real-world practical use case.
  - Takes less than 45px vertical height, keeping controls close to the image preview.

### 5. 🔬 Real-Time Radiometric Telemetry
Every pipeline execution computes real-time quantitative metrics:
- **Execution Latency:** Pipeline execution time in milliseconds ($< 15\text{ ms}$).
- **Active Layers:** Count of active vs. bypassed stages.
- **Color Space:** Real-time tensor format detector (RGB 24-bit, Grayscale 8-bit, Binary 1-bit).
- **Dynamic Range:** Actual minimum and maximum output intensity $[r_{\min}, r_{\max}]$.
- **PSNR Metric:** Peak Signal-to-Noise Ratio ($10 \log_{10} \frac{255^2}{\text{MSE}}$).
- **SSIM Index:** Structural Similarity Index measure.
- **Shannon Entropy:** Information density in bits per pixel ($-\sum p_i \log_2 p_i$).

---

## 🎬 Industry & Creative Workflow Presets

Test complete multi-stage pipelines in 1 click:

| Preset Name | Category | Pipeline Stages Included |
| :--- | :--- | :--- |
| **✨ Glamour Skin Retouch** | Photoshop Studio | Bilateral Filter $\rightarrow$ Smart Vibrance $\rightarrow$ Unsharp Masking $\rightarrow$ Radial Vignette |
| **🎨 Fine Art Pencil Sketch** | Creative FX | Hand-Drawn Pencil Sketch (Color Dodge) $\rightarrow$ Darkroom Optical Vignette |
| **🖍️ Graphic Cel Cartoon** | Creative FX | Multi-Pass Bilateral Flattening $\rightarrow$ Ink Outlines $\rightarrow$ 5-Level Color Posterization |
| **📷 Retro Sepia & Grain** | Film Studio | Antique Sepia Tint $\rightarrow$ Gaussian Film Grain Noise $\rightarrow$ Heavy Vignette Falloff |
| **🎬 Cinema Film Grade** | DaVinci Resolve | Contrast Stretch $\rightarrow$ von Kries White Balance $\rightarrow$ Lift/Gamma/Gain $\rightarrow$ HSI Vector |
| **📄 Document OCR Prep** | Document Imaging | Arithmetic Mean $\rightarrow$ High-Boost Sharpen $\rightarrow$ Otsu Binarization $\rightarrow$ Morph Opening |
| **📐 Structural Edge & Hough** | Computer Vision | Median Filter $\rightarrow$ Canny Edge Linking $\rightarrow$ Hough Transform Vector Lines |
| **🔬 Impulse Denoising** | Restoration | Salt & Pepper Noise $\rightarrow$ Contraharmonic Filter (Order Q) $\rightarrow$ Histogram Equalization |
| **🎥 Hollywood LUT & Fourier** | Color Grading | Cinematic 3D LUT Tone $\rightarrow$ 2D Fourier Gaussian Highpass Filter (FFT) |

---

## 📚 Detailed Sub-Documentation

For in-depth references, check the dedicated modular guides:
- 📐 **[Algorithms & Mathematical Catalog](algorithms/README.md):** Complete guide to all 66 algorithms, formulas, and mapped lab experiments.
- 🖥️ **[UI & Studio Architecture](ui/README.md):** Technical design guide for the sticky viewport, scrollable shelf, and dark studio design system.

---

## 🧪 Verification & Automated Test Suite

Pixel-Forge includes an automated end-to-end verification test suite covering all 66 algorithms, quantitative metrics, and presets:

```powershell
python tests/test_pipeline.py
```

**Test Results:**
```
=== Testing All DIP Algorithms in Registry ===
  [PASS] geo_sampling           (Spatial Sampling & Discretization)
  [PASS] geo_quantization       (Bit-Depth Quantization)
  [PASS] geo_affine             (Composite Affine Transformation)
  [PASS] geo_canvas_resize      (Canvas Resize & Letterbox Fitting)
  [PASS] geo_lens_distortion    (Optical Lens Distortion (Fisheye/Pincushion))
  [PASS] geo_perspective        (Perspective Keystone Quad Transform)
  [PASS] rad_negative           (Digital Photographic Negative)
  [PASS] rad_log                (Logarithmic Dynamic Range Compression)
  [PASS] rad_gamma              (Power-Law (Gamma) Correction)
  [PASS] rad_contrast_stretch   (Piecewise-Linear Contrast Stretching)
  [PASS] rad_bit_plane          (Bit-Plane Slicing)
  [PASS] rad_hist_eq            (Histogram Equalization)
  [PASS] rad_hist_spec          (Histogram Specification (Matching))
  [PASS] rad_brightness_contrast(Photoshop Brightness & Contrast)
  [PASS] rad_solarize           (Sabattier Darkroom Solarization)
  [PASS] rad_posterize          (Color Posterization Tonal Bands)
  [PASS] rest_gaussian_blur     (Gaussian Spatial Smoothing)
  [PASS] rest_motion_blur       (Directional Motion Blur)
  [PASS] rest_bilateral         (Bilateral Filter (Smart Blur / Skin Retouch))
  [PASS] rest_unsharp_mask      (Photoshop Unsharp Masking (USM))
  [PASS] col_vibrance           (Photoshop Vibrance & Smart Saturation)
  [PASS] col_sepia              (Antique Sepia Tone Tint)
  [PASS] col_split_toning       (DaVinci Resolve Split Toning)
  [PASS] seg_adaptive_thresh    (Local Adaptive Gaussian Binarization)
  [PASS] seg_hough_circles      (Hough Circle Transform)
  [PASS] seg_harris_corners     (Harris & Stephens Corner Detection)
  [PASS] art_pencil_sketch      (Hand-Drawn Pencil Sketch)
  [PASS] art_cartoonify         (Cel-Shaded Cartoonify Effect)
  [PASS] art_emboss             (3D Directional Bas-Relief Emboss)
  [PASS] art_vignette           (Radial Optical Lens Vignette)
  [PASS] art_tilt_shift         (Tilt-Shift Miniature Blur)
  ...
Result: 66 passed, 0 failed out of 66 algorithms (100% Pass Rate).
🎉 ALL TESTS PASSED SUCCESSFULLY!
```

---

## 📄 License
This project is open-source under the MIT License.
