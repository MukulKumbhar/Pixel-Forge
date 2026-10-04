# 🔬 Pixel-Forge: High-Performance Modular Image Processing Engine & Non-Destructive Studio

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/Computer_Vision-OpenCV%20%7C%20NumPy-5C3EE8.svg?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Test Suite](https://img.shields.io/badge/Test_Suite-66%2F66%20Passing-brightgreen.svg?style=flat-square)](tests/test_pipeline.py)
[![Architecture](https://img.shields.io/badge/Architecture-Non--Destructive%20DAG-orange.svg?style=flat-square)]()
[![Zero Flicker](https://img.shields.io/badge/Event_Loop-Zero--Flicker%20Native%20Callbacks-blueviolet.svg?style=flat-square)]()

**Pixel-Forge** is an open-source, high-performance, non-destructive Digital Image Processing (DIP) studio and computational vision workbench. Engineered with **Python, NumPy, OpenCV, and Streamlit**, it provides a real-time, node-graph inspired pipeline for chaining, inspecting, parameterizing, and serializing 66 mathematical image processing algorithms across 7 specialized vision domains with sub-15 millisecond execution latency.

---

## 📑 Table of Contents

1. [Introduction & Abstract](#-introduction--abstract)
2. [Problem Statement](#-problem-statement)
3. [Proposed Solution](#-proposed-solution)
4. [Core Methodology & Engineering Principles](#-core-methodology--engineering-principles)
5. [System Architecture](#-system-architecture)
6. [Interactive Studio & Dual-Viewport Canvas](#-interactive-studio--dual-viewport-canvas)
7. [Algorithmic Suite & Domain Classification](#-algorithmic-suite--domain-classification)
8. [Production Workflow Pipelines & Industry Presets](#-production-workflow-pipelines--industry-presets)
9. [Quantitative Diagnostics & Objective Telemetry](#-quantitative-diagnostics--objective-telemetry)
10. [Quickstart & Installation Guide](#-quickstart--installation-guide)
11. [Quality Assurance & Automated Test Suite](#-quality-assurance--automated-test-suite)
12. [Technical References & Citations](#-technical-references--citations)
13. [License & Maintainer](#-license--maintainer)

---

## 📖 Introduction & Abstract

Digital Image Processing operations have historically been bifurcated into two disparate environments:
1. **Academic Scripting Environments:** Standalone scripts that execute algorithms on isolated images, outputting static figures without interactive parameter feedback, multi-stage pipeline visibility, or non-destructive undo/redo capability.
2. **Proprietary Creative Desktop Applications (e.g. Adobe Photoshop, DaVinci Resolve):** Black-box commercial systems that provide graphical workflows but conceal underlying mathematical formulations, offer no contract-level tensor transparency, and prevent algorithmic customization.

**Pixel-Forge** unifies scientific computational transparency with professional digital content creation (DCC) ergonomics. Operating on a **non-destructive serial Directed Acyclic Graph (DAG) paradigm**, it enables researchers, computer vision engineers, and digital artists to construct multi-stage image processing pipelines, fine-tune continuous and discrete parameters via synchronized event callbacks, visualize instant comparative error heatmaps, and export reproducible pipeline recipes.

---

## ⚠️ Problem Statement

Modern visual computing workflows face four critical bottlenecks:

1. **Destructive In-Place Raster Mutation:**
   Standard OpenCV scripts typically follow an in-place mutation pattern (`img = cv2.filter2D(img, ...); img = cv2.threshold(img, ...)`). Modifying an upstream parameter requires a complete script restart and reload from storage, discarding intermediate computational states.
2. **Tensor Dimension & Color Space Mismatches:**
   Chaining disparate algorithms frequently triggers fatal matrix dimension errors (e.g. feeding an RGB $H \times W \times 3$ tensor into an Otsu binarizer or morphological boundary extractor that strictly mandates a 2D $H \times W$ single-channel matrix).
3. **Absence of Real-Time Objective Quality Telemetry:**
   Operators often tune filter parameters subjectively by eye without visibility into quantitative structural degradation (SSIM), information entropy, mean squared error (MSE), or peak signal-to-noise ratio (PSNR).
4. **Reactive UI State Desynchronization & Cascading Reruns:**
   Web-based reactive computational dashboards frequently suffer from widget state desynchronization, circular rerun ping-ponging, and frame flickering when multiple input controllers (sliders, steppers) attempt to bind to the same underlying parameter.

---

## 💡 Proposed Solution

Pixel-Forge resolves these challenges through a modular, contract-driven architecture:

- **Non-Destructive Layer Stack Execution:** The source frame is preserved immutably in memory. Each processing stage exists as an isolated transformation descriptor within an ordered serial stack, evaluated on-the-fly in milliseconds.
- **Contract-Based Auto-Converting Shims:** Every algorithm formally defines an `input_contract` (`ANY`, `RGB`, `GRAYSCALE`, `BINARY`) and an `output_contract`. The pipeline dynamically bridges tensor discrepancies via standard luminance weighting ($Y = 0.299R + 0.587G + 0.114B$) and thresholding, preventing runtime crashes.
- **Native Event-Driven Callbacks (Zero-Flicker):** Parameter mutations execute through native Streamlit `on_change` event hooks before the page reruns, updating pipeline states in-memory with zero cascading reruns or screen flickering.
- **Integrated Objective Telemetry HUD:** Synchronous computation of latency (ms), PSNR (dB), MSE, SSIM Index, and Shannon Entropy (bits/pixel) provides real-time scientific feedback.
- **Portable Pipeline Serialization:** Pipelines serialize to human-readable JSON recipes, guaranteeing 100% scientific reproducibility across workstations.

---

## ⚙️ Core Methodology & Engineering Principles

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SOURCE IMAGE INGESTION                          │
│     (Benchmark Synthetic Generators or Custom PNG/JPG/TIFF Upload)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     REACTIVE SESSION STATE MANAGER                     │
│   • active_layer_stack: [ {id, algo_key, params, auto_convert}, ... ]  │
│   • 15-state deep-copy undo / redo history stack                       │
│   • Native on_change callbacks executing prior to script evaluation    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   CONTRACT-BASED AUTO-CONVERT SHIMS                    │
│      Validates Tensor Shape & Color Space Before Every Filter:         │
│      • RGB (H, W, 3) ──[0.299R + 0.587G + 0.114B]──► Grayscale (H, W) │
│      • Grayscale (H, W) ────────[Otsu / Fixed]──────► Binary (H, W)    │
│      • Single-Channel (H, W) ────[cv2.cvtColor]─────► RGB (H, W, 3)    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   VECTORIZED MATHEMATICAL EXECUTION                    │
│   Pure NumPy & OpenCV array slicing (Zero UI coupling in algorithms/)  │
│   Stage 1 ──► Stage 2 ──► Stage 3 ──► ... ──► Stage N                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   REAL-TIME RADIOMETRIC TELEMETRY                      │
│   • End-to-End Latency (ms)     • Peak Signal-to-Noise Ratio (PSNR)    │
│   • Mean Squared Error (MSE)    • Structural Similarity Index (SSIM)   │
│   • Shannon Entropy (bits/pix)  • Multi-Channel Dynamic Range          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     RESPONSIVE FULL-SCREEN VIEWPORT                    │
│   • Side-by-Side Dual View      • Full Canvas Processed View           │
│   • Interactive Split Wipe      • Residual Difference Heatmap (JET)    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    STUDIO CONTROL SUITE (4 EXPANDERS)                  │
│   1. Filter Catalog & Algorithm Selector (Browse, Inspect & Add)      │
│   2. Active Filter Parameters & Stage Controls (Fine-Tuning Sliders)   │
│   3. Comparative Histograms (RGB vs Luminance Distributions)           │
│   4. Export Studio & Pipeline Recipe (PNG Export & JSON Serialization) │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🏛️ System Architecture

Pixel-Forge strictly decouples mathematical kernel computation, state orchestration, and user interface rendering:

```
Pixel-Forge/
├── app.py                     # Studio entry point: Orchestrates layout, state & execution
├── core/                      # Pipeline orchestration & diagnostic metrics
│   ├── pipeline.py            # Non-destructive layer executor & contract validator
│   ├── state_manager.py       # State lifecycle, 15-state undo/redo & workflow presets
│   └── metrics.py             # Quantitative diagnostics: PSNR, MSE, SSIM, Entropy, Histograms
├── algorithms/                # Pure mathematical functions (Zero UI dependencies)
│   ├── geometry.py            # Affine, spatial sampling, quantization, lens distortion, keystone
│   ├── radiometric.py         # Negative, log, gamma, contrast stretch, bit-planes, hist eq, posterize
│   ├── restoration.py         # Noise generators, mean, median, Gaussian blur, motion blur, bilateral, USM
│   ├── colorimetry.py         # HSI vector grading, CMYK, lift/gamma/gain, white balance, vibrance, sepia
│   ├── segmentation.py        # Sobel, Prewitt, Laplacian, Canny, Otsu, Hough lines/circles, adaptive thresh
│   ├── morphology.py          # Structuring elements, dilation, erosion, open, close, gradient, top/black-hat
│   ├── frequency.py           # 2D Fourier FFT magnitude spectrum, Butterworth & Gaussian filters
│   ├── artistic.py            # Pencil sketch, cartoonify, bas-relief emboss, radial vignette, tilt-shift
│   └── registry.py            # Central registry: 66 algorithm definitions across 7 domains
├── ui/                        # Presentation layer & responsive DCC design system
│   ├── styles.py              # Studio dark-mode design system & responsive CSS rules
│   ├── viewport.py            # Full-screen dual-viewport canvas & real-time telemetry HUD
│   ├── parameter_drawer.py    # Synchronized parameter sliders, hot-swapping & filter catalog
│   └── conflict_resolver.py   # State contract validator & auto-convert shims
├── project_guide/             # 📘 Comprehensive guide, algorithm mathematics & viva manual
│   └── README.md              # In-depth reference manual & top 25 viva voce defense Q&A
└── tests/
    └── test_pipeline.py       # Automated unit test suite verifying all 66 algorithms & presets
```

---

## 🖥️ Interactive Studio & Dual-Viewport Canvas

### Responsive Viewport Architecture
The display canvas implements dynamic responsive constraints (`max-height: 75vh !important; object-fit: contain !important;`), dynamically adapting across laptop screens, tablets, and 4K ultrawide monitors without component clipping or scrolling overlap.

### 4 Viewport Display Modes

| Viewport Mode | Technical Implementation | Practical Use Case |
| :--- | :--- | :--- |
| **Side-by-Side (Dual)** | Dual-column layout rendering Source and Processed tensors simultaneously. | Comparing raw documents against binarized text. |
| **Full Canvas (Processed)** | Full-width container rendering only the processed tensor at native resolution. | Inspecting subtle 3D LUT cinema film color grades. |
| **Interactive Split Wipe** | 1. Input/output aligned via bilinear interpolation.<br>2. Format harmonized.<br>3. Split column: $c = \lfloor \frac{\text{pos}}{100} \cdot W \rfloor$.<br>4. NumPy array slice concatenation with a $2\text{px}$ white boundary divider. | Sliding wipe across skin retouching to inspect before/after boundary. |
| **Difference Heatmap** | 1. Absolute pixel difference computed: $\Delta = \|I_{\text{source}} - I_{\text{proc}}\|$.<br>2. Luminance collapsed to single channel.<br>3. Colorized via OpenCV `COLORMAP_JET`. | Visualizing high-frequency edge loss after Gaussian or median blur. |

### Studio Control Suite (4 Dedicated Expanders)
1. **🎛️ Filter Catalog & Algorithm Selector:** Browse algorithms by domain, inspect mathematical definitions and real-world examples, and add stages to the active pipeline.
2. **⚙️ Active Filter Parameters & Stage Controls:** Active stage switcher, algorithm hot-swapping, auto-convert toggling, stage reset, and high-precision parameter sliders.
3. **📊 Comparative Image Histograms:** Dual comparative frequency distributions comparing input vs output across RGB multi-channel or composite luminance.
4. **💾 Export Studio & Pipeline Recipe:** Download full-resolution processed PNGs, export pipeline `.json` recipes, or import existing recipes for instant reconstruction.

---

## 🧮 Algorithmic Suite & Domain Classification

Pixel-Forge features **66 fully implemented mathematical algorithms** grouped into 7 technical computer vision domains:

```
Pixel-Forge Algorithm Master Suite (66 Algorithms)
├── 1. Geometric & Spatial Transformations (7)
│   ├── Spatial Sampling & Discretization
│   ├── Bit-Depth Quantization
│   ├── Composite Affine Transformation
│   ├── Canvas Scaling & Crop
│   ├── Arbitrary Angle Rotation
│   ├── Directional Shear Transform
│   └── Axial Cartesian Reflection (Flip)
├── 2. Radiometric & Dynamic Range Calibration (10)
│   ├── Photographic Negative
│   ├── Dynamic Range Logarithmic Compression
│   ├── Power-Law (Gamma) Contrast Correction
│   ├── Piecewise Linear Contrast Stretch
│   ├── Bit-Plane Decomposition & Slicing
│   ├── Cumulative Histogram Equalization
│   ├── Histogram Specification / Matching
│   ├── Linear Brightness & Contrast Calibration
│   ├── Sabattier Darkroom Solarization
│   └── Tonal Posterization Bands
├── 3. Stochastic Denoising & Image Restoration (10)
│   ├── Additive Gaussian Noise Injection
│   ├── Impulse Salt & Pepper Noise
│   ├── Multiplicative Speckle Noise
│   ├── Arithmetic Mean Spatial Filter
│   ├── Geometric Mean Non-Linear Filter
│   ├── Contraharmonic Mean Filter (Q-order)
│   ├── Order-Statistic Median Filter
│   ├── Alpha-Trimmed Mean Filter
│   ├── Min / Max / Midpoint Filter
│   └── 2D Gaussian Spatial Smoothing
├── 4. Chromatic Engine & Color Grading (9)
│   ├── 3D Cinematic LUT Color Transform
│   ├── Trigonometric HSI Color Vector Grading
│   ├── 3-Way Lift / Gamma / Gain Balancer
│   ├── von Kries Chromatic Adaptation (White Balance)
│   ├── Subtractive CMYK Decomposition
│   ├── Multi-Channel Linear RGB Mixer
│   ├── Photoshop Smart Vibrance
│   ├── Antique Photographic Sepia Print
│   └── DaVinci Resolve Split Toning
├── 5. Feature Extraction & Segmentation (10)
│   ├── Sobel Directional Gradient Filter
│   ├── Prewitt Compass Gradient Filter
│   ├── Laplacian Second-Derivative Operator
│   ├── Unsharp Masking & High-Boost Filter
│   ├── Multi-Stage Hysteresis Canny Edge Detector
│   ├── Global Fixed-Level Binarization
│   ├── Otsu Maximum Between-Class Variance Thresholding
│   ├── Optimal Iterative Mean Thresholding
│   ├── Parametric Hough Line Transform
│   └── Local Adaptive Gaussian Thresholding
├── 6. Morphological Mechanics & Frequency Domain (11)
│   ├── Morphological Dilation
│   ├── Morphological Erosion
│   ├── Morphological Opening
│   ├── Morphological Closing
│   ├── Morphological Gradient
│   ├── White Top-Hat Transform
│   ├── Black-Hat (Bottom-Hat) Transform
│   ├── Morphological Boundary Extraction
│   ├── Hit-or-Miss Structural Probe
│   ├── 2D Fourier (FFT) Magnitude Spectrum
│   └── 2D Frequency Domain Filtering (Ideal/Butterworth/Gaussian)
└── 7. Creative FX & Digital Content Creation (9)
    ├── Hand-Drawn Pencil Sketch (Color Dodge Blending)
    ├── Cel-Shaded Pop Art Cartoonify (Multi-Bilateral + Ink)
    ├── 3D Directional Bas-Relief Emboss
    ├── Radial Optical Lens Vignette
    ├── Tilt-Shift Miniature Depth Blur
    ├── Directional Motion Velocity Blur
    ├── Bilateral Filter (Skin Smoothing / Smart Blur)
    ├── Photoshop Unsharp Masking (USM with Noise Gate)
    └── Harris & Stephens Keypoint Corner Detection
```

---

## 🎬 Production Workflow Pipelines & Industry Presets

Pixel-Forge includes 9 pre-engineered industrial workflow presets that demonstrate complex multi-stage pipeline chaining:

| Preset Name | Target Industry | Chained Stage Pipeline | Primary Visual Objective |
| :--- | :--- | :--- | :--- |
| **✨ Photoshop Studio Retouch** | Portrait Fashion Photography | Bilateral Filter $\to$ Smart Vibrance $\to$ Unsharp Masking $\to$ Optical Vignette | Smooths facial skin while preserving sharp eyelashes, boost skin vibrance, and focus lighting. |
| **🎨 Fine Art Pencil Drawing** | Digital Art & Illustration | Grayscale Invert $\to$ Wide Gaussian Blur $\to$ Color Dodge Division $\to$ Vignette | Simulates authentic graphite hand-drawn pencil artwork with edge-hugging pencil strokes. |
| **🖍️ Pop Art Cel-Shading** | Graphic Novels & Comic Art | 3-Pass Bilateral Color Flattening $\to$ Adaptive Ink Contour Extraction $\to$ Color Posterize | Creates high-contrast comic book artwork with bold black ink contours and flat cel colors. |
| **📷 Retro Antique Sepia** | Archival & Vintage Film | Linear Sepia Matrix $\to$ Additive Gaussian Grain Noise $\to$ Radial Vignette | Reproduces early 20th-century antique silver print photographs with warm sepia and analog grain. |
| **🎬 Cinema Film Grade** | Hollywood Post-Production | Dynamic Contrast Stretch $\to$ von Kries White Balance $\to$ Lift/Gamma/Gain $\to$ HSI Vector | Cinematic color science pipeline performing dynamic expansion, temperature tint, and 3-way color balance. |
| **📄 Document Binarization** | OCR Pre-Processing & Archival | Arithmetic Mean Filter $\to$ High-Boost Sharpening $\to$ Otsu Thresholding $\to$ Morph Opening | Cleans degraded document scans under uneven lighting, sharpening faded text and eliminating ink bleed. |
| **📐 Structural Edge & Hough** | Robotics & Autonomous Vision | Median Denoise $\to$ Canny Multi-Stage Hysteresis $\to$ Parametric Hough Line Voting | Detects linear structural geometry, road lane markings, and architectural boundaries. |
| **🔬 Impulse Noise Restoration**| Forensic Restoration | Salt & Pepper Inoculation $\to$ Contraharmonic Mean Filter ($Q=1.5$) $\to$ Hist Equalization | Demonstrates mathematical recovery of corrupted sensor rasters using non-linear order statistics. |
| **🎥 Hollywood 3D LUT & FFT**| VFX & Feature Film | 3D Teal & Orange LUT $\to$ 2D Fourier Gaussian Lowpass $\to$ High-Boost Spatial Filter | Applies cinematic cross-channel color transformation combined with 2D frequency domain filtering. |

---

## 📊 Quantitative Diagnostics & Objective Telemetry

Pixel-Forge computes 5 objective mathematical metrics in real time between the raw input and processed output:

### 1. Mean Squared Error (MSE)
$$\text{MSE} = \frac{1}{M \cdot N \cdot C} \sum_{c=1}^C \sum_{x=1}^M \sum_{y=1}^N \Big[I_{\text{raw}}(x, y, c) - I_{\text{proc}}(x, y, c)\Big]^2$$
*Quantifies average squared pixel intensity discrepancy. Lower is closer to the original.*

### 2. Peak Signal-to-Noise Ratio (PSNR)
$$\text{PSNR} = 20 \cdot \log_{10}\left(\frac{255}{\sqrt{\text{MSE}}}\right) \quad (\text{in decibels, dB})$$
*Measures reconstruction fidelity against degradation. Values $> 30\text{ dB}$ indicate excellent structural retention.*

### 3. Structural Similarity Index (SSIM)
$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
*Evaluates perceptual similarity based on luminance, contrast, and structural correlation. Range: $[-1.0, 1.0]$.*

### 4. Shannon Information Entropy
$$H = -\sum_{k=0}^{L-1} p(r_k) \log_2 p(r_k) \quad (\text{in bits per pixel})$$
*Quantifies statistical randomness and informational complexity. Flat histograms approach $8.0\text{ bits/pixel}$.*

### 5. Execution Latency
High-resolution performance counter measurement ($t_{\text{end}} - t_{\text{start}}$) in milliseconds, reflecting real-time pipeline performance.

---

## 🚀 Quickstart & Installation Guide

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Git

### 1. Clone Repository
```bash
git clone https://github.com/MukulKumbhar/Pixel-Forge.git
cd Pixel-Forge
```

### 2. Create Virtual Environment & Install Dependencies
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Launch Studio
```bash
streamlit run app.py
```
The application will launch in your browser at `http://localhost:8501`.

---

## 🧪 Quality Assurance & Automated Test Suite

Pixel-Forge includes an automated regression test suite covering all 66 algorithms, quantitative telemetry calculations, and built-in industry workflow presets:

```bash
python tests/test_pipeline.py
```

### Test Coverage Highlights:
- **66/66 Algorithm Invocations:** Validates numerical stability, output tensor types (`uint8`), channel consistency, and shape integrity.
- **Quantitative Metrics Validation:** Verifies MSE, PSNR, SSIM, and Shannon Entropy calculations against ground truth arrays.
- **9/9 Industrial Preset Executions:** Validates multi-stage pipeline chaining, auto-convert shims, and conflict-free execution.

```
Result: 66 passed, 0 failed out of 66 algorithms.
Result: 9 passed, 0 failed out of 9 workflow presets.
🎉 ALL TESTS PASSED SUCCESSFULLY!
```

---

## 📚 Technical References & Citations

1. **Gonzalez, R. C., & Woods, R. E. (2018).** *Digital Image Processing* (4th ed.). Pearson.
2. **Tomasi, C., & Manduchi, R. (1998).** *Bilateral Filtering for Gray and Color Images.* IEEE International Conference on Computer Vision (ICCV), 839–846.
3. **Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).** *Image quality assessment: from error visibility to structural similarity.* IEEE Transactions on Image Processing, 13(4), 600–612.
4. **Canny, J. (1986).** *A computational approach to edge detection.* IEEE Transactions on Pattern Analysis and Machine Intelligence, (6), 679–698.
5. **Otsu, N. (1979).** *A threshold selection method from gray-level histograms.* IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62–66.
6. **Harris, C., & Stephens, M. (1988).** *A combined corner and edge detector.* Alvey Vision Conference, 15(50), 147–151.
7. **Duda, R. O., & Hart, P. E. (1972).** *Use of the Hough transformation to detect lines and curves in pictures.* Communications of the ACM, 15(1), 11–15.
8. **Brown, D. C. (1971).** *Close-range camera calibration.* Photogrammetric Engineering, 37(8), 855–866.

---

## 📄 License & Maintainer

- **Author:** Mukul Kumbhar
- **Repository:** [https://github.com/MukulKumbhar/Pixel-Forge](https://github.com/MukulKumbhar/Pixel-Forge)
- **License:** Released under the [MIT License](LICENSE). Open-source for academic, commercial, and research use.
