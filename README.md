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
2. [Dual Scope: Academic Foundation + Creative DCC Suite](#-dual-scope-academic-foundation--creative-dcc-suite)
3. [End-to-End System Workflow & Working Order](#-end-to-end-system-workflow--working-order)
4. [Deep-Dive: Viewport Implementation & Canvas Scaling System](#-deep-dive-viewport-implementation--canvas-scaling-system)
5. [Academic Lab Experiment Mapping (Exp 01 — 10)](#-academic-lab-experiment-mapping-exp-01--10)
6. [Creative Studio & Photoshop Replica Suite](#-creative-studio--photoshop-replica-suite)
7. [Studio Architecture & Layout](#-studio-architecture--layout)
8. [Industry & Creative Workflow Presets](#-industry--creative-workflow-presets)
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

The interactive studio will open in your browser at `http://localhost:8501`.

---

## 🌟 Dual Scope: Academic Foundation + Creative DCC Suite

Pixel-Forge was engineered to provide **both**:
1. **Academic Rigor:** 100% adherence to standard university DIP syllabi and laboratory experiments (sampling, quantization, point transformations, CDF histogram equalization, spatial filtering, trigonometric HSI colorimetry, differential edge operators, Hough transforms, morphological mechanics, and 2D Fourier FFT).
2. **Professional Creative Power:** A full Photoshop / DaVinci Resolve replica toolset (Unsharp Masking, Bilateral skin retouching, Pencil Sketch, Cel-Shaded Cartoonify, Bas-Relief Emboss, Radial Vignette, Tilt-Shift Miniature, Fisheye Lens Distortion, Keystone Perspective Warp, Smart Vibrance, Antique Sepia, Split-Toning, Adaptive Thresholding, Hough Circles, and Harris Corners).

---

## 🔄 End-to-End System Workflow & Working Order

The studio executes a synchronized, non-destructive multi-stage pipeline on every user interaction:

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. SOURCE FRAME INGESTION                                              │
│    • Benchmark sample selection or custom image upload (PNG/JPG/TIFF)  │
│    • Matrix normalization to uint8 (H × W × C)                         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ 2. PIPELINE LAYER ORCHESTRATION                                        │
│    • Sidebar: Load Preset or Add Stages across 7 Domains (66 tools)    │
│    • Reorder (▲/▼), Bypass (Active checkbox), Delete (✕), Settings (⚙️)│
│    • Undo / Redo history tracking with automatic 15-state stack        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ 3. CONFLICT RESOLUTION & SHIM BRIDGING                                 │
│    • Pre-flight format validator detects contract mismatches           │
│    • Auto-convert shims automatically bridge RGB ↔ Grayscale ↔ Binary  │
│    • Prevents crashes when feeding RGB tensors into single-channel ops │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ 4. MATHEMATICAL KERNEL EXECUTION                                       │
│    • Pure NumPy & OpenCV vectorized execution across active stages     │
│    • Zero UI coupling; isolated inside algorithms/ directory           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ 5. TELEMETRY & QUANTITATIVE DIAGNOSTICS                                │
│    • Real-time metric calculations: Latency (ms), Active count, PSNR,  │
│      MSE, SSIM Index, and Shannon Entropy (bits/pixel)                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ 6. RESPONSIVE FULL-SCREEN VIEWPORT RENDERING                           │
│    • Side-by-Side Dual View | Full Canvas | Split Wipe | Error Heatmap │
│    • Parameter Drawer with Micro-Info Card (What it does + Example)    │
│    • Synchronized Comparative Histograms & Export Studio (PNG / JSON)  │
└────────────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Execution Order:
1. **Input Normalization:** Image loaded into a NumPy `uint8` ndarray with standardized RGB channel ordering.
2. **State Contract Validation:** As frames flow through each stage, `core/pipeline.py` verifies the input format against `input_contract` (`RGB`, `GRAYSCALE`, `BINARY`, or `ANY`).
3. **Auto-Convert Shims:** If an algorithm requires `GRAYSCALE` but receives `RGB`, the pipeline automatically converts via standard luminance weights ($Y = 0.299R + 0.587G + 0.114B$) without user intervention.
4. **Parameter Injection:** The stage executes with current slider/numeric parameters from session state.
5. **Real-Time Telemetry:** Metrics calculated synchronously between original source and final raster.
6. **Viewport Update:** Instantaneous rendering without page reload or layout overlapping.

---

## 🖥️ Deep-Dive: Viewport Implementation & Canvas Scaling System

### 1. Canvas Scale Presets Architecture
In earlier iterations, image canvas heights were controlled via discrete pixel height presets:
- **`Compact (400px)`:** Minimized height for small laptop screens or inspection alongside multiple expanded drawers.
- **`Comfortable (520px)`:** Balanced height maintaining controls in immediate visual proximity.
- **`Spacious (640px)`:** Expanded workspace prioritizing detail examination.
- **`Ultra Detail (760px)`:** Maximum height for high-resolution 4K and ultrawide workstations.

#### How It Was Implemented Under the Hood:
- **Dynamic CSS Injection:** The studio injected dynamic CSS rules where `__PREVIEW_HEIGHT__` was string-substituted with the selected pixel integer:
  ```css
  [data-testid="stImage"] img {
      max-height: __PREVIEW_HEIGHT__px !important;
      width: auto !important;
      object-fit: contain !important;
  }
  ```
- **Streamlit State Binding:** The dropdown selection was tracked in `st.session_state["preview_height"]`. When changed, it triggered a state update and rerun.

#### What Actually Happened Behind the Scenes & Why Fixed Presets Were Replaced:
- **The Issue:** Hardcoding fixed pixel `max-height` constraints created layout conflicts across different device screen resolutions and aspect ratios (e.g. portrait documents vs landscape panoramas). Fixed containers frequently overlapped surrounding Streamlit divs or clipped parameter sliders below.
- **The Solution (Native Full-Screen Responsive Viewport):**
  We eliminated manual height dropdowns in favor of a clean, responsive layout:
  ```css
  [data-testid="stImage"] {
      display: flex !important;
      justify-content: center !important;
      width: 100% !important;
  }
  [data-testid="stImage"] img {
      max-width: 100% !important;
      height: auto !important;
      max-height: 75vh !important;
      object-fit: contain !important;
      border-radius: 10px !important;
  }
  ```
  - **Dynamic Viewport Height (`75vh`):** Images automatically scale to fill available viewport space up to 75% of viewport height, guaranteeing that parameter controls below remain immediately visible without scrolling.
  - **`use_container_width=True`:** Preserves 100% raster aspect ratio with zero distortion, zero pixel squishing, and zero component overlap.

---

### 2. Viewport Display Modes (Under the Hood)

| Viewport Mode | Mathematical / Implementation Mechanism | Practical Example |
| :--- | :--- | :--- |
| **Side-by-Side (Dual)** | Dual Streamlit column flex layout (`st.columns(2)`) rendering Source and Processed tensors concurrently. | Comparing raw document scan against Otsu binarized text. |
| **Full Canvas (Processed)** | Full-width container rendering only the processed tensor at native resolution. | Inspecting subtle 3D LUT cinema film color grading. |
| **Interactive Split Wipe** | 1. Input/output aligned via bilinear interpolation (`cv2.resize`).<br>2. Format harmonized (Grayscale promoted to 3-channel if compared with RGB).<br>3. Horizontal split column calculated: $c = \lfloor \frac{\text{pos}}{100} \cdot W \rfloor$.<br>4. NumPy slice concatenation: `canvas[:, :c] = source[:, :c]`.<br>5. 2px white dividing line drawn at column $c$. | Sliding wipe across skin retouching to see instant before/after boundary. |
| **Difference Heatmap** | 1. Absolute pixel difference computed: $\Delta = \|I_{\text{source}} - I_{\text{proc}}\|$.<br>2. Luminance collapsed to single channel.<br>3. Pseudo-color thermal mapping applied via `cv2.applyColorMap(Δ, cv2.COLORMAP_JET)`. | Visualizing high-frequency edge loss after Gaussian or median blur. |

---

## 🎓 Academic Lab Experiment Mapping (Exp 01 — 10)

| Lab Exp | Topic & Syllabus Focus | Implemented Algorithms | Mathematical Formulation | Practical Example |
| :---: | :--- | :--- | :--- | :--- |
| **Exp 01** | Image I/O & Channel Properties | Synthetic Generator, Channel Slices | $I(x,y) \in [0, 255]^C$ | Inspecting R, G, B channel separation on test charts |
| **Exp 02** | Pixel Access & Intensity | Luminance extraction, RGB Scopes | $Y = 0.299R + 0.587G + 0.114B$ | Isolating perceptual brightness channel from color |
| **Exp 03** | Sampling & Quantization | Nearest/Bilinear/Bicubic, Bit-Depth (8 to 1 bit) | $I_q(x,y) = \lfloor \frac{I(x,y)}{256/2^b} \rfloor \cdot \frac{255}{2^b - 1}$ | Simulating 2-bit retro arcade graphics or low-bandwidth transmission |
| **Exp 04** | Arithmetic Operations | Add, Subtract, Blend, Difference | $I_{\text{blend}} = \alpha I_1 + (1-\alpha) I_2$ | Double exposure photography and background subtraction |
| **Exp 05** | Point Transformations | Negative, Log, Gamma, Stretch, Bit-Planes | $s = c \log(1+r)$, $s = c r^\gamma$, $s = 255 - r$ | Expanding dark shadow detail in underexposed astrophotography (Gamma $\gamma < 1$) |
| **Exp 06** | Histogram Processing | CDF Equalization, Histogram Specification | $s_k = T(r_k) = (L-1) \sum_{j=0}^k p_r(r_j)$ | Enhancing contrast in low-contrast medical X-rays |
| **Exp 07** | Spatial Filtering | Mean, Median, Min/Max, Contraharmonic | $f(x,y) = \text{median}\{g(s,t)\}$, $Q$-order filter | Eliminating salt & pepper impulse noise without blurring edges |
| **Exp 08** | Color Models | RGB $\leftrightarrow$ HSI, CMYK, White Balance | $H = \arccos\left(\frac{\frac{1}{2}[(R-G)+(R-B)]}{\sqrt{(R-G)^2+(R-B)(G-B)}}\right)$ | Correcting yellow tungsten color cast via von Kries chromatic adaptation |
| **Exp 09** | Edge Detection | Sobel, Prewitt, Laplacian, Canny | $G = \sqrt{G_x^2 + G_y^2}$, $\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$ | Industrial barcode detection and structural boundary localization |
| **Exp 10** | Hough Transform | Hough Lines ($\rho = x\cos\theta + y\sin\theta$), Hough Circles | Accumulator voting in parameter space | Detecting lane lines in autonomous navigation systems |

---

## 🎨 Creative Studio & Photoshop Replica Suite

Beyond standard syllabus experiments, Pixel-Forge equips you with professional DCC capabilities:

1. **Photoshop Unsharp Masking (USM):** Crisp definition enhancement using high-pass unsharp mask subtraction with adjustable threshold noise gate.
2. **Bilateral Skin Retouching (Smart Blur):** Non-linear spatial/range Gaussian filtering that smooths skin pores while preserving sharp eyelashes and lips.
3. **Graphite Pencil Sketch:** Blends inverted Gaussian blur with grayscale raster via Color Dodge division for authentic hand-drawn graphite art.
4. **Cel-Shaded Pop Art Cartoon:** Multi-pass bilateral color quantization overlaid with adaptive Laplacian ink outline strokes.
5. **3D Bas-Relief Emboss:** Directional lighting convolution simulating stamped commemorative coins and carved stone relief.
6. **Radial Optical Vignette:** Quadratic falloff spotlight shading simulating vintage Leica 35mm f/1.4 street photography lenses.
7. **Tilt-Shift Miniature Photography:** Defocus blur gradient outside a selectable sharp focal band turning real cityscapes into toy dioramas.
8. **Optical Lens Distortion:** Radial polynomial distortion ($r' = r(1 + k_1 r^2 + k_2 r^4)$) simulating GoPro fisheye lenses or correcting architecture.
9. **Perspective Keystone Quad Transform:** Four-corner planar homography for straightening angled document scans and tilted buildings.
10. **Smart Vibrance:** Non-linear saturation boosting that protects skin tones from harsh oversaturation.
11. **Antique Sepia Tone:** Professional $3 \times 3$ photographic silver print toning matrix.
12. **DaVinci Resolve Split Toning:** Complementary shadow and highlight chromatic split grading.
13. **Local Adaptive Binarization:** Local window Gaussian thresholding for documents under uneven desk lamps or casting shadows.
14. **Hough Circle Transform:** Automated gradient voting for circular coins, pupils, and iris localization.
15. **Harris Corner Detection:** Second-moment auto-correlation matrix ($R = \det(M) - k \cdot \text{trace}(M)^2$) for keypoint feature matching.

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
│   ├── README.md              # 📖 UI layout guide: Responsive canvas, parameter drawer & CSS theme
│   ├── styles.py              # Studio dark-mode DCC design system & responsive CSS rules
│   ├── viewport.py            # Full-screen dual-viewport canvas & real-time telemetry HUD
│   ├── parameter_drawer.py    # Synchronized sliders, hot-swapping & filter micro-info cards
│   └── conflict_resolver.py   # State contract validator & auto-convert shims (RGB ↔ Gray ↔ Binary)
├── assets/
│   ├── generate_samples.py    # Synthetic standard reference image generator
│   └── sample_images/         # Standard benchmark images (Chroma, Siemens Star, OCR, Shapes)
├── tests/
│   └── test_pipeline.py       # Automated verification test suite (66 algorithms & 9 presets)
└── requirements.txt           # Project dependencies
```

---

## 🎬 Industry & Creative Workflow Presets

Test complete multi-stage pipelines in 1 click from the sidebar preset selector:

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
