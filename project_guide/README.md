# 📘 Pixel-Forge — Comprehensive Project Guide, Theoretical Foundations & Viva Voce Manual

> **Project Author:** Mukul Kumbhar  
> **Repository:** [MukulKumbhar/Pixel-Forge](https://github.com/MukulKumbhar/Pixel-Forge)  
> **Project Scope:** Modular Non-Destructive Digital Image Processing Studio, Computational Vision Engine & Professional Creative DCC Suite  
> **Target Audience:** Project Defense, Viva Voce Examination, Technical Interviews, and Academic Evaluation  

---

## 📑 Table of Contents

1. [Executive Project Overview & Working Order](#1-executive-project-overview--working-order)
2. [End-to-End System Execution Flow (Start to Finish)](#2-end-to-end-system-execution-flow-start-to-finish)
3. [Academic Syllabus Alignment vs Extra Industry-Grade Features](#3-academic-syllabus-alignment-vs-extra-industry-grade-features)
4. [Deep-Dive: Mathematical Formulations & Logic of Extra / Advanced Algorithms](#4-deep-dive-mathematical-formulations--logic-of-extra--advanced-algorithms)
5. [Complete Master Catalog: All 66 Implemented Algorithms](#5-complete-master-catalog-all-66-implemented-algorithms)
6. [Core Engineering Decisions & Architectural Innovations](#6-core-engineering-decisions--architectural-innovations)
7. [Quantitative Telemetry & Objective Image Quality Assessment](#7-quantitative-telemetry--objective-image-quality-assessment)
8. [Top 25 Viva Voce Defense Questions & Model Answers](#8-top-25-viva-voce-defense-questions--model-answers)

---

## 1. Executive Project Overview & Working Order

### 1.1 What is Pixel-Forge?
**Pixel-Forge** is an interactive, high-performance, non-destructive Digital Image Processing (DIP) studio and computational vision workbench. Built with **Python, NumPy, OpenCV, and Streamlit**, it enables users to chain, reorder, parameterize, and inspect image transformations in real time with sub-15 millisecond execution latency.

Unlike conventional image processing scripts that destructively overwrite raster pixel data in-place, Pixel-Forge operates on a **Directed Acyclic Graph (DAG) layer stack paradigm** similar to Adobe Photoshop, Adobe After Effects, and DaVinci Resolve. The raw source frame is immutably preserved in memory, while each active filter acts as an isolated mathematical transform node evaluated sequentially.

### 1.2 High-Level System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SOURCE IMAGE INGESTION                          │
│     (Benchmark Synthetic Generators or Custom PNG/JPG/TIFF Upload)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     STREAMLIT REACTIVE STATE ENGINE                    │
│   • active_layer_stack: [ {id, algo_key, params, auto_convert}, ... ]  │
│   • undo_stack / future_stack (15-state deep-copy ring buffer)         │
│   • Native on_change event callbacks (zero-flicker lifecycle)         │
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
│   Executes pure NumPy / OpenCV kernel for each active stage:           │
│   Stage 1 ──► Stage 2 ──► Stage 3 ──► ... ──► Stage N                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   REAL-TIME RADIOMETRIC TELEMETRY                      │
│   • Execution Latency (ms)      • Peak Signal-to-Noise Ratio (PSNR)    │
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

## 2. End-to-End System Execution Flow (Start to Finish)

When a user opens the application, adjusts a slider, or adds a filter, the system executes through the following deterministic phases:

### Phase 1: Image Ingestion & Normalization
1. The user selects a built-in benchmark image (Lena, Mandrill, Cameraman, Einstein, Peppers, or Synthetic Geometry Grid) or uploads a custom raster file.
2. The image is parsed via `cv2.imdecode` or PIL, converted to a standardized `uint8` NumPy ndarray with standard RGB channel layout $(H, W, 3)$ or 2D Grayscale $(H, W)$, and stored in `raw_base_image`.
3. The raw image is immutable and cached in memory.

### Phase 2: User Action & Native `on_change` Callback Execution
1. The user interacts with a widget (e.g., changes a slider, toggles an active checkbox, selects an algorithm).
2. **Crucial Architectural Detail:** Streamlit's native `on_change` callback (e.g., `_on_param_change`) fires **before** the main script body reruns.
3. The callback updates the targeted parameter inside `st.session_state.active_layer_stack[stage_index]["params"][param_name]` in-memory.
4. **No manual `st.rerun()` is triggered**, eliminating all widget state desynchronization, infinite ping-pong loops, and screen flickering.

### Phase 3: Non-Destructive Pipeline Execution (`execute_pipeline`)
1. The pipeline starts with a copy of `raw_base_image`.
2. It iterates through the list of active layers in `active_layer_stack`.
3. For each layer:
   - **Bypass Check:** If `layer["enabled"] == False`, the stage is skipped with status `"bypassed"`.
   - **Contract Validation:** `detect_state(current_tensor)` determines if the current tensor is `RGB`, `GRAYSCALE`, or `BINARY`.
   - **Auto-Conversion Shim:** If the algorithm requires `GRAYSCALE` but receives `RGB`, the shim automatically calculates luminance:
     $$Y = 0.299R + 0.587G + 0.114B$$
   - **Kernel Invocation:** The mathematical function is called dynamically from `algorithms/` with the unpacked parameters:
     $$\text{output\_tensor} = f(\text{current\_tensor}, **\text{params})$$
   - **Safety Assertion:** The output tensor is validated for finite numerical bounds, valid shape, and cast to `uint8`.
4. A structured `PipelineExecutionResult` object is returned, containing the final output image, per-stage execution times, input/output tensor shapes, and state transitions.

### Phase 4: Radiometric Telemetry & Quality Benchmarking
The engine compares `raw_base_image` against `processed_image` across 5 quantitative metrics:
1. **Execution Latency:** High-resolution timer (`time.perf_counter`) captures end-to-end processing time in milliseconds.
2. **Mean Squared Error (MSE):** Cumulative squared pixel variance.
3. **Peak Signal-to-Noise Ratio (PSNR):** Logarithmic decibel ratio of peak signal power to noise power.
4. **Structural Similarity Index (SSIM):** Perceptual structural preservation score evaluating luminance, contrast, and structure.
5. **Shannon Information Entropy:** Information density measured in bits per pixel.

### Phase 5: Responsive Dual-Viewport Canvas Rendering
The viewport renders the processed raster based on the selected mode:
- **Side-by-Side (Dual):** Direct spatial comparison of source and processed output.
- **Full Canvas (Processed):** High-definition unobstructed output.
- **Interactive Split Wipe:** Real-time horizontal sliding reveal bar dividing input and output.
- **Difference Heatmap:** Colorized residual error map ($\|I_{\text{source}} - I_{\text{proc}}\|$) mapped through the JET false-color colormap.

### Phase 6: Expander Controls & Recipe Export
The user can inspect comparative histograms, fine-tune parameters, browse new algorithms from the catalog, or serialize the entire pipeline configuration into a portable `.json` recipe file for reproducible experiments.

---

## 3. Academic Syllabus Alignment vs Extra Industry-Grade Features

A central highlight of Pixel-Forge is that it bridges the gap between **academic textbook theory** (standard syllabus units) and **modern industry digital content creation (DCC)** (Photoshop / DaVinci Resolve creative workflows).

### 3.1 Academic Syllabus Breakdown (Units 1 — 6)

| Academic Unit | Standard Syllabus Curriculum | Implemented Algorithms in Pixel-Forge |
| :--- | :--- | :--- |
| **Unit 1: Fundamentals & Geometry** | Image representation, sampling, quantization, resolution, affine geometry, interpolation. | `geo_sampling`, `geo_quantization`, `geo_affine`, `geo_canvas_resize`, `geo_rotation`, `geo_shear`, `geo_flip` |
| **Unit 2: Radiometry & Point Ops** | Brightness, contrast, negative, log transform, power-law (gamma), contrast stretching, histogram equalization, bit-plane slicing. | `rad_negative`, `rad_log_transform`, `rad_gamma`, `rad_contrast_stretch`, `rad_bit_plane`, `rad_hist_eq`, `rad_hist_match`, `rad_brightness_contrast`, `rad_solarize`, `rad_posterize` |
| **Unit 3: Image Restoration & Filtering** | Degradation models, noise injection, spatial filtering (mean, median, contraharmonic, alpha-trimmed, order statistics), Gaussian blur. | `rest_noise_gaussian`, `rest_noise_sp`, `rest_noise_speckle`, `rest_arithmetic_mean`, `rest_geometric_mean`, `rest_contraharmonic`, `rest_median`, `rest_alpha_trimmed`, `rest_order_stats`, `rest_gaussian_blur` |
| **Unit 4: Color Image Processing** | Color models (RGB, HSI, CMYK), color slicing, pseudocolor, multi-channel color transformations. | `col_hsi_vector`, `col_cmyk`, `col_rgb_matrix`, `col_white_balance`, `col_vibrance`, `col_sepia` |
| **Unit 5: Segmentation & Edge Detection** | Point/line/edge detection, first derivative (Sobel, Prewitt), second derivative (Laplacian), Canny edge detector, thresholding (Otsu, global, adaptive), Hough line transform. | `seg_sobel_balancer`, `seg_prewitt`, `seg_laplacian`, `seg_high_boost`, `seg_canny`, `seg_manual_thresh`, `seg_otsu`, `seg_iterative_thresh`, `seg_hough_lines`, `seg_adaptive_thresh` |
| **Unit 6: Morphological Processing & Frequency Domain** | Erosion, dilation, opening, closing, morphological gradient, top-hat, black-hat, boundary extraction, hit-or-miss, 2D Discrete Fourier Transform (DFT), ideal/Butterworth/Gaussian lowpass and highpass filters. | `morph_dilate`, `morph_erode`, `morph_open`, `morph_close`, `morph_gradient`, `morph_tophat`, `morph_blackhat`, `morph_boundary`, `morph_hit_and_miss`, `freq_spectrum`, `freq_filter` |

---

### 3.2 What is EXTRA (Advanced Industry-Grade Features)?

Standard college syllabi typically stop at basic grayscale filters and simple edge detection. Pixel-Forge includes **15+ advanced, industry-grade algorithms** modeled after production tools in Adobe Photoshop, Adobe Lightroom, and DaVinci Resolve:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   ADVANCED & INDUSTRY-GRADE EXTENSIONS                 │
│                                                                        │
│   🎨 Creative FX & Photoshop Suite:                                    │
│      • Bilateral Skin Retouching (Edge-Preserving Surface Blur)        │
│      • Photoshop Unsharp Masking (USM) with Noise Coring Threshold    │
│      • Graphite Pencil Sketch (Color Dodge Blending)                   │
│      • Cel-Shaded Pop Art Cartoonify (Multi-Pass Bilateral + Ink)      │
│      • 3D Directional Bas-Relief Emboss                                │
│      • Radial Optical Vignette (Cosine-Fourth Falloff)                 │
│      • Tilt-Shift Miniature Defocus Gradient                           │
│                                                                        │
│   🎬 Cinematic Color Grading & DaVinci Resolve Engine:                 │
│      • 3D Cinematic LUT Trilinear Interpolation                        │
│      • Three-Way Lift / Gamma / Gain Color Grading                     │
│      • DaVinci Resolve Split Toning (Shadow / Highlight Chromatic Split)│
│      • von Kries Chromatic Adaptation & Color Temperature / Tint       │
│      • Photoshop Smart Vibrance (Skin-Tone Protecting Saturation)      │
│                                                                        │
│   📐 Advanced Optical & Computer Vision Geometry:                      │
│      • Radial Optical Lens Distortion (Barrel & Fisheye Polynomial)    │
│      • 4-Corner Planar Perspective Homography (Keystone Rectification) │
│      • Parametric Hough Circle Transform (Accumulator Gradient Voting) │
│      • Harris & Stephens Second-Moment Keypoint Corner Detection       │
│      • Directional Motion Blur (Velocity Vector Point Spread Function) │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Deep-Dive: Mathematical Formulations & Logic of Extra / Advanced Algorithms

Examiners frequently ask for the mathematical derivations of algorithms that go beyond the basic textbook. Below are the precise mathematical foundations, algorithmic steps, and rationales for all extra algorithms:

---

### 4.1 Bilateral Filter (Photoshop "Smart Blur" / Glamour Skin Retouching)
- **Problem with Gaussian Blur:** Standard Gaussian blur averages all neighboring pixels uniformly based solely on spatial distance. This removes noise but severely destroys sharp edge boundaries (e.g. facial features, eyelashes, lips).
- **The Bilateral Solution:** The bilateral filter weights neighboring pixels by **both** spatial closeness (domain kernel) **and** photometric radiometric similarity (range kernel).

#### Mathematical Formulation:
$$I^{\text{filtered}}(x) = \frac{1}{W_p} \sum_{x_i \in \Omega} I(x_i) \cdot g_s(\|x_i - x\|) \cdot g_r(\|I(x_i) - I(x)\|)$$

Where the spatial Gaussian $g_s$ is:
$$g_s(\|x_i - x\|) = \exp\left(-\frac{\|x_i - x\|^2}{2\sigma_s^2}\right)$$

And the radiometric range Gaussian $g_r$ is:
$$g_r(\|I(x_i) - I(x)\|) = \exp\left(-\frac{\|I(x_i) - I(x)\|^2}{2\sigma_r^2}\right)$$

And $W_p$ is the normalization factor ensuring energy preservation:
$$W_p = \sum_{x_i \in \Omega} g_s(\|x_i - x\|) \cdot g_r(\|I(x_i) - I(x)\|)$$

- **How It Works:** When the filter encounters a flat area (skin texture), $\|I(x_i) - I(x)\|$ is small, so $g_r \approx 1$ and it behaves like standard Gaussian smoothing. When it crosses an edge (e.g. skin to eye), $\|I(x_i) - I(x)\|$ is large, $g_r \to 0$, and pixels across the edge are excluded from the average.
- **Parameters:** `d` (neighborhood diameter), `sigma_color` ($\sigma_r$, tonal boundary tolerance), `sigma_space` ($\sigma_s$, spatial reach).

---

### 4.2 Photoshop Unsharp Masking (USM) with Noise Coring Gate
- **Academic Unsharp Masking:** $f_{\text{sharp}} = f + k(f - f_{\text{smooth}})$. The weakness is that it amplifies high-frequency camera noise and film grain.
- **Photoshop USM Enhancement:** Introduces a non-linear **Threshold Noise Gate (Coring)**:
  $$\Delta(x, y) = f(x, y) - f_{\text{Gaussian}}(x, y)$$
  $$\Delta_{\text{gated}}(x, y) = \begin{cases} \Delta(x, y), & \text{if } |\Delta(x, y)| \ge \text{threshold} \\ 0, & \text{if } |\Delta(x, y)| < \text{threshold} \end{cases}$$
  $$f_{\text{out}}(x, y) = \text{clip}\Big(f(x, y) + \text{amount} \cdot \Delta_{\text{gated}}(x, y), 0, 255\Big)$$
- **Logic:** Microscopic noise variations below `threshold` are ignored, while true structural contours are crisply boosted by `amount`.

---

### 4.3 Graphite Pencil Sketch (Color Dodge Blending)
- **Algorithm:** Replicates authentic hand-drawn graphite pencil artwork using a 3-step optical simulation:
  1. **Grayscale Desaturation:** Convert source image to luminance:
     $$I_{\text{gray}} = 0.299R + 0.587G + 0.114B$$
  2. **Negative Inversion & Gaussian Blur:** Invert pixel values and apply wide Gaussian smoothing:
     $$I_{\text{inv}} = 255 - I_{\text{gray}}$$
     $$I_{\text{blur}} = \mathcal{G}_{\sigma}(I_{\text{inv}})$$
  3. **Color Dodge Mathematical Blending:**
     $$I_{\text{sketch}}(x, y) = \text{clip}\left( \frac{I_{\text{gray}}(x, y) \cdot 256}{255 - I_{\text{blur}}(x, y) + 1}, 0, 255 \right)$$
- **Why this works:** When $I_{\text{blur}}$ is close to $I_{\text{inv}}$ (smooth areas), the denominator $(255 - I_{\text{blur}})$ equals $I_{\text{gray}}$, and the division yields pure white ($255$). Only where high-frequency edge discontinuities exist does $I_{\text{blur}}$ diverge from $I_{\text{inv}}$, producing dark graphite pencil lines.

---

### 4.4 Cel-Shaded Pop Art Cartoonify
- **Algorithm:** Combines non-linear color quantization with edge extraction:
  1. **Multi-Pass Bilateral Smoothing:** Applies $N$ repeated passes of bilateral filtering to homogenize color gradients while keeping sharp silhouette boundaries.
  2. **Median Blur & Adaptive Thresholding:** Computes an adaptive threshold on a median-filtered luminance channel to generate high-contrast comic-book ink lines:
     $$E(x, y) = \text{AdaptiveThreshold}(\text{Median}(I_{\text{gray}}), \text{ksize}=9, C=2)$$
  3. **Bitwise Compositing:** Merges the color-flattened image with the black ink contour mask via bitwise AND:
     $$I_{\text{cartoon}} = I_{\text{bilateral}} \ \& \ \text{stack}_3(E)$$

---

### 4.5 3D Directional Bas-Relief Emboss
- **Algorithm:** Convolves the image with an asymmetric directional spatial kernel:
  $$K_{\text{emboss}} = \begin{bmatrix} -2 & -1 & 0 \\ -1 & 1 & 1 \\ 0 & 1 & 2 \end{bmatrix}$$
- **Luminance Bias Offset:** Since convolution output contains negative gradients, a $128$ midpoint gray bias is added:
  $$I_{\text{emboss}}(x, y) = \text{clip}\Big((I * K_{\text{emboss}})(x, y) + 128, 0, 255\Big)$$
- **Effect:** Flat areas turn neutral gray ($128$), while edges facing top-left become highlights ($> 128$) and edges facing bottom-right become shadows ($< 128$), creating the optical illusion of stamped commemorative metal coins.

---

### 4.6 Radial Optical Lens Vignette
- **Physical Principle:** Real camera lenses exhibit natural illuminance falloff toward frame edges described by the **$\cos^4(\theta)$ optical law** (natural vignetting).
- **Mathematical Model:**
  $$d(x, y) = \sqrt{\left(\frac{x - x_{\text{center}}}{W/2}\right)^2 + \left(\frac{y - y_{\text{center}}}{H/2}\right)^2}$$
  $$V(x, y) = 1.0 - \text{intensity} \cdot \left(\frac{d(x, y)}{\text{radius} \cdot \sqrt{2}}\right)^2$$
  $$I_{\text{vignette}}(x, y) = \text{clip}\Big(I(x, y) \cdot V(x, y), 0, 255\Big)$$
- **Result:** Smoothly darkens the peripheral perimeter of the frame, naturally drawing the viewer's eye to the central photographic subject.

---

### 4.7 Tilt-Shift Miniature Photography (Optical Diorama Blur)
- **Principle:** Simulates the ultra-shallow depth of field of a large-format tilt-shift bellows camera lens, making life-size buildings and cars look like toy miniature dioramas.
- **Mathematical Model:**
  1. Defines an in-focus horizontal focal plane band $[y_1, y_2]$ centered at $y_{\text{focus}}$ with width $h_{\text{band}}$.
  2. Computes normalized optical distance from the focal band:
     $$\delta(y) = \max\left(0, \frac{|y - y_{\text{focus}}| - h_{\text{band}}/2}{H/2}\right)$$
  3. Pre-computes a deep Gaussian blur of the image: $I_{\text{defocus}} = \mathcal{G}_{\sigma}(I)$.
  4. Blends the sharp image with the defocused image using quadratic weight $\alpha(y) = \min(1.0, \delta(y)^2)$:
     $$I_{\text{tilt-shift}}(x, y) = (1 - \alpha(y)) \cdot I(x, y) + \alpha(y) \cdot I_{\text{defocus}}(x, y)$$

---

### 4.8 3D Cinematic LUT (Lookup Table) Trilinear Interpolation
- **Problem:** 1D tone curves can modify individual RGB channels, but cannot simulate complex non-linear cross-channel color grading (e.g. film chemical emulation, Teal & Orange split).
- **Solution:** A 3D color cube lattice of size $N \times N \times N$ (e.g. $17 \times 17 \times 17$ or $33 \times 33 \times 33$).
- **Algorithm:** For any input color $(R, G, B) \in [0, 255]^3$:
  1. Normalize input coordinates to lattice indices:
     $$u = \frac{R}{255} \cdot (N - 1), \quad v = \frac{G}{255} \cdot (N - 1), \quad w = \frac{B}{255} \cdot (N - 1)$$
  2. Locate the 8 surrounding bounding vertices of the enclosing unit cube:
     $$c_{000}, c_{001}, c_{010}, c_{011}, c_{100}, c_{101}, c_{110}, c_{111}$$
  3. Perform trilinear interpolation along fractional offsets $(\Delta u, \Delta v, \Delta w)$:
     $$\text{Color}_{\text{graded}} = \sum_{i,j,k \in \{0,1\}} c_{ijk} (1 - |\Delta u - i|)(1 - |\Delta v - j|)(1 - |\Delta w - k|)$$

---

### 4.9 DaVinci Resolve Lift / Gamma / Gain (Three-Way Color Balancer)
- **Industry Standard:** Used by professional colorists to adjust shadows, midtones, and highlights independently:
  - **Lift ($L$):** Offsets the black point (shadows) without clipping highlights.
  - **Gain ($G$):** Scales the white point (highlights) pivoting around pure black.
  - **Gamma ($\gamma$):** Non-linear power-law inflection targeting midtone exposure.
- **Mathematical Formulation:**
  For normalized input $x \in [0, 1]$:
  $$x_{\text{lift}} = x \cdot (1 - L) + L$$
  $$x_{\text{gain}} = x_{\text{lift}} \cdot G$$
  $$x_{\text{final}} = (x_{\text{gain}})^{1/\gamma}$$

---

### 4.10 von Kries Chromatic Adaptation & White Balance
- **Physical Law:** Human vision perceives colors consistently under varying illumination (color constancy). The von Kries diagonal model scales cone responses independently:
  $$\begin{bmatrix} L' \\ M' \\ S' \end{bmatrix} = \begin{bmatrix} L_w / L_s & 0 & 0 \\ 0 & M_w / M_s & 0 \\ 0 & 0 & S_w / S_s \end{bmatrix} \begin{bmatrix} L \\ M \\ S \end{bmatrix}$$
- **Implementation in Pixel-Forge:**
  Simulates color temperature adjustments along the Planckian locus (warm orange $\leftrightarrow$ cool blue) and tint adjustments along the daylight locus (green $\leftrightarrow$ magenta):
  $$R' = R \cdot (1 + T), \quad B' = B \cdot (1 - T)$$
  $$G' = G \cdot (1 + \text{Tint}), \quad R' = R' \cdot (1 - \text{Tint}/2), \quad B' = B' \cdot (1 - \text{Tint}/2)$$

---

### 4.11 Photoshop Smart Vibrance
- **Problem with Saturation:** Standard saturation scales the HSV/HSI saturation channel linearly for all pixels. When photos contain human portraits, skin tones become unnaturally orange, blotchy, and burnt.
- **The Vibrance Logic:** Non-linearly boosts under-saturated pixels while protecting already-saturated colors and warm skin hues:
  $$S_{\text{new}} = S + \text{vibrance} \cdot (1 - S) \cdot S \cdot (1 - \text{SkinWeight})$$
  Where $\text{SkinWeight}$ is computed by checking if the pixel's hue falls in the human skin range ($H \in [0^\circ, 35^\circ]$).

---

### 4.12 Radial Optical Lens Distortion (Barrel & Fisheye)
- **Brown-Conrady Polynomial Model:**
  $$r = \sqrt{\left(\frac{x - x_c}{W/2}\right)^2 + \left(\frac{y - y_c}{H/2}\right)^2}$$
  $$r_{\text{distorted}} = r \cdot (1 + k_1 r^2 + k_2 r^4)$$
  $$x_{\text{distorted}} = x_c + (x - x_c) \cdot \frac{r_{\text{distorted}}}{r}, \quad y_{\text{distorted}} = y_c + (y - y_c) \cdot \frac{r_{\text{distorted}}}{r}$$
- **Interpretation:**
  - $k_1 > 0$: **Pincushion Distortion** (corners pulled outward, common in telephoto zoom lenses).
  - $k_1 < 0$: **Barrel Distortion** (spherical bulging, common in wide-angle and GoPro action cameras).

---

### 4.13 Planar Perspective Homography (Keystone Correction)
- **Problem:** Capturing documents or building architecture from an angle introduces keystoning (perspective trapezoidal distortion).
- **Mathematical Solution:** Maps four source quadrilateral corners $(x_i, y_i)$ to an aligned rectangular bounding box $(x'_i, y'_i)$ via a $3 \times 3$ projective homography matrix $H$:
  $$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
- **Estimation:** Solved via Singular Value Decomposition (SVD) on Direct Linear Transformation (DLT) equations from the 4 point correspondences.

---

### 4.14 Harris & Stephens Corner Detection
- **Mathematical Formulation:** Evaluates local autocorrelation via the Second-Moment Matrix $M$:
  $$M(x, y) = \sum_{(u, v) \in W} w(u, v) \begin{bmatrix} I_x^2(u, v) & I_x I_y(u, v) \\ I_x I_y(u, v) & I_y^2(u, v) \end{bmatrix}$$
- **Corner Response Measure $R$:**
  $$R = \det(M) - k \cdot (\text{trace}(M))^2$$
  Where $\det(M) = \lambda_1 \lambda_2$ and $\text{trace}(M) = \lambda_1 + \lambda_2$.
- **Classification:**
  - $R > 0$: **Corner Region** ($\lambda_1$ and $\lambda_2$ are both large).
  - $R < 0$: **Edge Region** ($\lambda_1 \gg \lambda_2$ or vice versa).
  - $|R| \approx 0$: **Flat Uniform Region** ($\lambda_1 \approx \lambda_2 \approx 0$).

---

## 5. Complete Master Catalog: All 66 Implemented Algorithms

Below is the exhaustive mathematical inventory of all 66 algorithms across all 7 domains:

### Domain 1: Geometric & Spatial Transformations (7 Algorithms)
| Key | Name | Input $\to$ Output | Mathematical Expression | Complexity |
| :--- | :--- | :---: | :--- | :---: |
| `geo_sampling` | Spatial Sampling & Discretization | ANY $\to$ SAME | $f_d[x, y] = f(x \cdot \Delta x, y \cdot \Delta y)$ | $\mathcal{O}(H \cdot W)$ |
| `geo_quantization` | Bit-Depth Quantization | ANY $\to$ SAME | $q = \lfloor r / (256/2^k) \rfloor \cdot (255 / (2^k - 1))$ | $\mathcal{O}(H \cdot W)$ |
| `geo_affine` | Composite Affine Transform | ANY $\to$ SAME | $\mathbf{x}' = \mathbf{A}\mathbf{x} + \mathbf{t}$ | $\mathcal{O}(H \cdot W)$ |
| `geo_canvas_resize` | Canvas Scaling & Crop | ANY $\to$ SAME | Bilinear / Bicubic Resampling | $\mathcal{O}(H' \cdot W')$ |
| `geo_rotation` | Arbitrary Angle Rotation | ANY $\to$ SAME | $R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$ | $\mathcal{O}(H \cdot W)$ |
| `geo_shear` | Directional Shear Transform | ANY $\to$ SAME | $x' = x + s_x y, \quad y' = y + s_y x$ | $\mathcal{O}(H \cdot W)$ |
| `geo_flip` | Axial Cartesian Reflection | ANY $\to$ SAME | $x' = W - 1 - x, \quad y' = H - 1 - y$ | $\mathcal{O}(H \cdot W)$ |

### Domain 2: Radiometric & Dynamic Range Calibration (10 Algorithms)
| Key | Name | Input $\to$ Output | Mathematical Expression | Complexity |
| :--- | :--- | :---: | :--- | :---: |
| `rad_negative` | Photographic Negative | ANY $\to$ SAME | $s = (L - 1) - r$ | $\mathcal{O}(H \cdot W)$ |
| `rad_log_transform` | Dynamic Range Logarithm | ANY $\to$ SAME | $s = c \log(1 + r)$ | $\mathcal{O}(H \cdot W)$ |
| `rad_gamma` | Power-Law (Gamma) Correction | ANY $\to$ SAME | $s = c \cdot r^\gamma$ | $\mathcal{O}(H \cdot W)$ |
| `rad_contrast_stretch`| Piecewise Linear Contrast Stretch | ANY $\to$ SAME | 3-Segment Linear Interpolation | $\mathcal{O}(H \cdot W)$ |
| `rad_bit_plane` | Bit-Plane Slicing (0 to 7) | GRAY $\to$ GRAY | $b_k(x, y) = (I(x, y) \gg k) \ \& \ 1$ | $\mathcal{O}(H \cdot W)$ |
| `rad_hist_eq` | Cumulative Histogram Equalization | ANY $\to$ SAME | $s_k = (L - 1) \sum_{j=0}^k p_r(r_j)$ | $\mathcal{O}(H \cdot W)$ |
| `rad_hist_match` | Histogram Specification / Matching | ANY $\to$ SAME | $z = G^{-1}(T(r))$ | $\mathcal{O}(H \cdot W)$ |
| `rad_brightness_contrast` | Photoshop Linear Brightness/Contrast | ANY $\to$ SAME | $s = \alpha \cdot r + \beta$ | $\mathcal{O}(H \cdot W)$ |
| `rad_solarize` | Sabattier Darkroom Solarization | ANY $\to$ SAME | $s = 4r(1 - r/255)$ | $\mathcal{O}(H \cdot W)$ |
| `rad_posterize` | Tonal Posterization Bands | ANY $\to$ SAME | Quantized Uniform Bins | $\mathcal{O}(H \cdot W)$ |

### Domain 3: Stochastic Denoising & Image Restoration (10 Algorithms)
| Key | Name | Input $\to$ Output | Mathematical Expression | Complexity |
| :--- | :--- | :---: | :--- | :---: |
| `rest_noise_gaussian`| Additive Gaussian Noise Injection | ANY $\to$ SAME | $g(x,y) = f(x,y) + \eta \sim \mathcal{N}(\mu, \sigma^2)$ | $\mathcal{O}(H \cdot W)$ |
| `rest_noise_sp` | Impulse Salt & Pepper Noise | ANY $\to$ SAME | $P(\text{salt}) = P(\text{pepper}) = p$ | $\mathcal{O}(H \cdot W)$ |
| `rest_noise_speckle` | Multiplicative Speckle Noise | ANY $\to$ SAME | $g(x,y) = f(x,y) \cdot (1 + \eta)$ | $\mathcal{O}(H \cdot W)$ |
| `rest_arithmetic_mean`| Arithmetic Mean Neighborhood Filter | ANY $\to$ SAME | $f(x,y) = \frac{1}{mn}\sum_{(s,t)\in S_{xy}} g(s,t)$ | $\mathcal{O}(H \cdot W \cdot k^2)$ |
| `rest_geometric_mean`| Geometric Mean Non-Linear Filter | ANY $\to$ SAME | $f(x,y) = \left[\prod_{(s,t)\in S_{xy}} g(s,t)\right]^{\frac{1}{mn}}$ | $\mathcal{O}(H \cdot W \cdot k^2)$ |
| `rest_contraharmonic`| Contraharmonic Mean Filter ($Q$) | ANY $\to$ SAME | $f(x,y) = \frac{\sum g(s,t)^{Q+1}}{\sum g(s,t)^Q}$ | $\mathcal{O}(H \cdot W \cdot k^2)$ |
| `rest_median` | Order-Statistic Median Filter | ANY $\to$ SAME | $f(x,y) = \text{median}_{(s,t)\in S_{xy}}\{g(s,t)\}$ | $\mathcal{O}(H \cdot W \cdot k^2)$ |
| `rest_alpha_trimmed` | Alpha-Trimmed Mean Filter | ANY $\to$ SAME | Trims $d/2$ extremes, averages remaining | $\mathcal{O}(H \cdot W \cdot k^2 \log k)$ |
| `rest_order_stats` | Min / Max / Midpoint Filter | ANY $\to$ SAME | $\frac{1}{2}[\max(g) + \min(g)]$ | $\mathcal{O}(H \cdot W \cdot k^2)$ |
| `rest_gaussian_blur` | 2D Gaussian Spatial Smoothing | ANY $\to$ SAME | $G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$ | $\mathcal{O}(H \cdot W \cdot k)$ |

### Domain 4: Chromatic Engine & Color Grading (9 Algorithms)
| Key | Name | Input $\to$ Output | Mathematical Expression | Complexity |
| :--- | :--- | :---: | :--- | :---: |
| `col_3d_lut` | 3D Cinematic LUT Color Transform | RGB $\to$ RGB | Trilinear Color Cube Interpolation | $\mathcal{O}(H \cdot W)$ |
| `col_hsi_vector` | Trigonometric HSI Color Vector Grading| RGB $\to$ RGB | RGB $\leftrightarrow$ HSI Non-linear Mapping | $\mathcal{O}(H \cdot W)$ |
| `col_lift_gamma_gain`| 3-Way Lift / Gamma / Gain Balancer | RGB $\to$ RGB | $(x(1-L) + L \cdot G)^{1/\gamma}$ | $\mathcal{O}(H \cdot W)$ |
| `col_white_balance` | von Kries Chromatic Adaptation | RGB $\to$ RGB | Diagonal Scaling Matrix | $\mathcal{O}(H \cdot W)$ |
| `col_cmyk` | Subtractive CMYK Decomposition | RGB $\to$ RGB | $K = 1 - \max(R,G,B), C = \frac{1-R-K}{1-K}$ | $\mathcal{O}(H \cdot W)$ |
| `col_rgb_matrix` | Linear $3 \times 3$ RGB Channel Mixer | RGB $\to$ RGB | $\mathbf{c}' = \mathbf{M}_{3\times3} \mathbf{c}$ | $\mathcal{O}(H \cdot W)$ |
| `col_vibrance` | Photoshop Smart Vibrance | RGB $\to$ RGB | $S + V(1-S)S(1-\text{SkinWeight})$ | $\mathcal{O}(H \cdot W)$ |
| `col_sepia` | Antique Photographic Sepia Print | RGB $\to$ RGB | Eastman Kodak Sepia Matrix | $\mathcal{O}(H \cdot W)$ |
| `col_split_toning` | DaVinci Resolve Split Toning | RGB $\to$ RGB | Complementary Shadow/Highlight Split | $\mathcal{O}(H \cdot W)$ |

### Domain 5: Feature Extraction & Segmentation (10 Algorithms)
| Key | Name | Input $\to$ Output | Mathematical Expression | Complexity |
| :--- | :--- | :---: | :--- | :---: |
| `seg_sobel_balancer` | Sobel Directional Gradient Filter | ANY $\to$ GRAY | $G = \sqrt{G_x^2 + G_y^2}, \theta = \arctan(G_y/G_x)$ | $\mathcal{O}(H \cdot W)$ |
| `seg_prewitt` | Prewitt Compass Gradient Filter | ANY $\to$ GRAY | $3 \times 3$ Uniform Differential Kernel | $\mathcal{O}(H \cdot W)$ |
| `seg_laplacian` | Laplacian Second-Derivative Operator | ANY $\to$ GRAY | $\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$ | $\mathcal{O}(H \cdot W)$ |
| `seg_high_boost` | Unsharp Masking & High-Boost Filter | ANY $\to$ SAME | $f_{\text{hb}} = A \cdot f - f_{\text{lowpass}}$ | $\mathcal{O}(H \cdot W)$ |
| `seg_canny` | Multi-Stage Hysteresis Canny Edge | ANY $\to$ GRAY | Gradient $\to$ Non-Max $\to$ Double Threshold | $\mathcal{O}(H \cdot W)$ |
| `seg_manual_thresh` | Global Fixed-Level Binarization | ANY $\to$ BINARY | $g(x,y) = 255 \text{ if } f \ge T \text{ else } 0$ | $\mathcal{O}(H \cdot W)$ |
| `seg_otsu` | Otsu Maximum Between-Class Variance | ANY $\to$ BINARY | $\sigma_B^2(t) = \omega_0\omega_1(\mu_0 - \mu_1)^2$ | $\mathcal{O}(H \cdot W + 256)$ |
| `seg_iterative_thresh`| Optimal Iterative Mean Thresholding | ANY $\to$ BINARY | $T_{k+1} = \frac{1}{2}(\mu_{\text{bg}} + \mu_{\text{fg}})$ | $\mathcal{O}(K \cdot H \cdot W)$ |
| `seg_hough_lines` | Parametric Hough Line Transform | ANY $\to$ SAME | $\rho = x \cos\theta + y \sin\theta$ Accumulator | $\mathcal{O}(N_{\text{edges}} \cdot N_\theta)$ |
| `seg_adaptive_thresh`| Local Adaptive Gaussian Thresholding | ANY $\to$ BINARY | $T(x,y) = \text{Mean}(S_{xy}) - C$ | $\mathcal{O}(H \cdot W)$ |

### Domain 6: Morphological Mechanics & Frequency Domain (11 Algorithms)
| Key | Name | Input $\to$ Output | Mathematical Expression | Complexity |
| :--- | :--- | :---: | :--- | :---: |
| `morph_dilate` | Morphological Dilation | ANY $\to$ SAME | $A \oplus B = \{z \mid (\hat{B})_z \cap A \neq \emptyset\}$ | $\mathcal{O}(H \cdot W \cdot k^2)$ |
| `morph_erode` | Morphological Erosion | ANY $\to$ SAME | $A \ominus B = \{z \mid (B)_z \subseteq A\}$ | $\mathcal{O}(H \cdot W \cdot k^2)$ |
| `morph_open` | Morphological Opening | ANY $\to$ SAME | $A \circ B = (A \ominus B) \oplus B$ | $\mathcal{O}(H \cdot W \cdot k^2)$ |
| `morph_close` | Morphological Closing | ANY $\to$ SAME | $A \bullet B = (A \oplus B) \ominus B$ | $\mathcal{O}(H \cdot W \cdot k^2)$ |
| `morph_gradient` | Morphological Gradient | ANY $\to$ SAME | $\text{Grad}(A) = (A \oplus B) - (A \ominus B)$ | $\mathcal{O}(H \cdot W \cdot k^2)$ |
| `morph_tophat` | White Top-Hat Transform | ANY $\to$ SAME | $\text{TopHat}(A) = A - (A \circ B)$ | $\mathcal{O}(H \cdot W \cdot k^2)$ |
| `morph_blackhat` | Black-Hat Transform | ANY $\to$ SAME | $\text{BlackHat}(A) = (A \bullet B) - A$ | $\mathcal{O}(H \cdot W \cdot k^2)$ |
| `morph_boundary` | Boundary Extraction Operator | ANY $\to$ SAME | $\beta(A) = A - (A \ominus B)$ | $\mathcal{O}(H \cdot W \cdot k^2)$ |
| `morph_hit_and_miss` | Hit-or-Miss Structural Probe | ANY $\to$ BINARY | $A \circledast B = (A \ominus B_1) \cap (A^c \ominus B_2)$ | $\mathcal{O}(H \cdot W)$ |
| `freq_spectrum` | 2D Fourier Magnitude Spectrum | ANY $\to$ GRAY | $|F(u,v)| = \sqrt{R^2(u,v) + I^2(u,v)}$ | $\mathcal{O}(N^2 \log N)$ |
| `freq_filter` | 2D Frequency Filtering (FFT) | ANY $\to$ SAME | $G(u,v) = H(u,v) \cdot F(u,v)$ | $\mathcal{O}(N^2 \log N)$ |

### Domain 7: Creative FX & Photoshop Studio (9 Algorithms)
| Key | Name | Input $\to$ Output | Mathematical Expression | Complexity |
| :--- | :--- | :---: | :--- | :---: |
| `art_pencil_sketch` | Hand-Drawn Pencil Sketch | ANY $\to$ SAME | Color Dodge Division on Inverted Blur | $\mathcal{O}(H \cdot W)$ |
| `art_cartoonify` | Cel-Shaded Cartoonify Effect | ANY $\to$ SAME | Multi-Bilateral + Adaptive Ink Lines | $\mathcal{O}(H \cdot W \cdot d)$ |
| `art_emboss` | 3D Directional Bas-Relief Emboss | ANY $\to$ SAME | $(I * K_{\text{emboss}}) + 128$ | $\mathcal{O}(H \cdot W)$ |
| `art_vignette` | Radial Optical Lens Vignette | ANY $\to$ SAME | $I \cdot (1 - \text{intensity} \cdot (d/R)^2)$ | $\mathcal{O}(H \cdot W)$ |
| `art_tilt_shift` | Tilt-Shift Miniature Depth Blur | ANY $\to$ SAME | In-Focus Band + Quadratic Defocus Ramp | $\mathcal{O}(H \cdot W)$ |
| `rest_motion_blur` | Directional Motion Velocity Blur | ANY $\to$ SAME | Linear Point Spread Function (PSF) | $\mathcal{O}(H \cdot W \cdot k)$ |
| `rest_bilateral` | Bilateral Filter (Skin Retouch) | ANY $\to$ SAME | Domain $\times$ Range Gaussian Kernels | $\mathcal{O}(H \cdot W \cdot d^2)$ |
| `rest_unsharp_mask` | Photoshop Unsharp Masking (USM) | ANY $\to$ SAME | $f + \alpha \cdot \text{Gate}(f - \mathcal{G}(f))$ | $\mathcal{O}(H \cdot W)$ |
| `seg_harris_corners`| Harris & Stephens Corner Detection | ANY $\to$ SAME | $R = \det(M) - k \cdot (\text{trace}(M))^2$ | $\mathcal{O}(H \cdot W)$ |

---

## 6. Core Engineering Decisions & Architectural Innovations

### 6.1 Why a Non-Destructive Serial Stack instead of In-Place Mutation?
- **Destructive Workflow (Traditional):** `img = filter1(img); img = filter2(img)`. When the user adjusts a slider on `filter1`, the original information is already lost, forcing a complete reload from disk.
- **Non-Destructive Workflow (Pixel-Forge):** The pipeline maintains `active_layer_stack` as a serial JSON configuration. On every adjustment, the pipeline re-executes the active stages from the clean immutable `raw_base_image`. Stages can be reordered ($\mathbf{\Delta}$/$\mathbf{\nabla}$), bypassed, or deleted without corrupting underlying raster buffers.

### 6.2 Contract-Based Auto-Converting Shims
- **The Problem:** In a multi-stage pipeline, `Stage 1` might output a 3-channel RGB image (e.g. Bilateral Retouch), but `Stage 2` might be an edge operator that strictly expects a 1-channel Grayscale matrix (e.g. Canny or Otsu). In naive implementations, this causes Python index errors (`IndexError: too many indices for array`).
- **The Solution:** Each algorithm in `algorithms/registry.py` declares an `input_contract` (`ANY`, `RGB`, `GRAYSCALE`, `BINARY`) and an `output_contract`. Before executing a stage, `core/conflict_resolver.py` checks the incoming tensor shape. If a contract mismatch exists and `auto_convert` is enabled, the shim automatically applies standard luminance weights without crashing or bothering the user.

### 6.3 Native `on_change` Callbacks (Eliminating Infinite Rerun Loops)
- **The Issue:** Early versions rendered both a slider and a number input for the same parameter and called `st.rerun()` whenever values changed. Because updating one widget left its sibling with stale session state, the two widgets ping-ponged endlessly at 60 FPS, causing severe screen flickering.
- **The Fix:** Migrated to a single high-precision `st.slider` per parameter bound directly to an `on_change=_on_param_change` callback. Streamlit executes the callback **prior** to the main script rerun, updating the layer state in-memory with zero cascading reruns.

### 6.4 Responsive Full-Screen Canvas (`75vh`)
- Replaced rigid, hardcoded pixel height dropdowns (`400px`, `520px`, `640px`) with modern responsive CSS viewport scaling (`max-height: 75vh !important; object-fit: contain !important;`). The canvas dynamically adapts to laptops, tablets, and 4K ultrawide monitors without component clipping or scrolling overlap.

---

## 7. Quantitative Telemetry & Objective Image Quality Assessment

Pixel-Forge embeds a real-time scientific telemetry HUD that continuously quantifies degradation and fidelity between the raw input and processed output:

### 1. Mean Squared Error (MSE)
Cumulative squared error between source $I_1$ and processed raster $I_2$:
$$\text{MSE} = \frac{1}{M \cdot N \cdot C} \sum_{c=1}^C \sum_{x=1}^M \sum_{y=1}^N \Big[I_1(x, y, c) - I_2(x, y, c)\Big]^2$$
- Ideal value: $0.0$ (identical images). Larger values indicate greater geometric or photometric alteration.

### 2. Peak Signal-to-Noise Ratio (PSNR)
Logarithmic measure of reconstruction quality in decibels (dB):
$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right) = 20 \cdot \log_{10}\left(\frac{255}{\sqrt{\text{MSE}}}\right)$$
- Values $> 30\text{ dB}$ represent excellent visual fidelity. Values $< 20\text{ dB}$ indicate heavy degradation or intentional artistic transformation.

### 3. Structural Similarity Index (SSIM)
Evaluates perceptual degradation based on human visual system (HVS) characteristics across luminance ($l$), contrast ($c$), and structure ($s$):
$$\text{SSIM}(x, y) = [l(x, y)]^\alpha \cdot [c(x, y)]^\beta \cdot [s(x, y)]^\gamma = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
- Range: $[-1.0, 1.0]$. A score of $1.0$ indicates identical structural definition.

### 4. Shannon Information Entropy
Measures statistical information density (randomness/texture content) in bits per pixel:
$$H = -\sum_{k=0}^{L-1} p(r_k) \log_2 p(r_k)$$
Where $p(r_k)$ is the normalized histogram probability of gray level $r_k$.
- High entropy ($\approx 7.5 - 8.0\text{ bits/pixel}$) signifies rich, complex textural detail. Low entropy signifies uniform or over-quantized regions.

---

## 8. Top 25 Viva Voce Defense Questions & Model Answers

### Q1: What is the primary architectural difference between Pixel-Forge and simple OpenCV script collections?
> **Answer:** Pixel-Forge implements a **non-destructive Directed Acyclic Graph (DAG) layer stack engine**. Traditional scripts mutate images in-place, losing source data on the first operation. Pixel-Forge preserves the raw input in an immutable memory buffer, evaluates serial stages with contract-based type checking, calculates multi-channel quantitative telemetry, and supports 15-state undo/redo history and JSON recipe serialization.

### Q2: Why is the Bilateral filter called "edge-preserving"?
> **Answer:** Unlike standard Gaussian blur which only measures spatial Euclidean distance $g_s(\|x_i - x\|)$, the bilateral filter incorporates a radiometric range Gaussian $g_r(\|I(x_i) - I(x)\|)$. When the filter window straddles a high-contrast edge, the photometric difference $|I(x_i) - I(x)|$ is large, driving $g_r \to 0$. Pixels on the opposite side of the boundary are assigned zero weight and excluded from the blur average, leaving the edge razor-sharp.

### Q3: How does the Graphite Pencil Sketch algorithm work mathematically?
> **Answer:** It uses a 3-step color dodge division technique:
> 1. Desaturate to luminance ($I_{\text{gray}}$).
> 2. Invert and Gaussian blur ($I_{\text{blur}} = \mathcal{G}(255 - I_{\text{gray}})$).
> 3. Perform Color Dodge: $I_{\text{sketch}} = \text{clip}\left(\frac{I_{\text{gray}} \cdot 256}{255 - I_{\text{blur}} + 1}, 0, 255\right)$. Flat regions cancel out to pure white ($255$), while high-frequency edge gradients become crisp graphite lines.

### Q4: What is the purpose of Auto-Convert Shims?
> **Answer:** To prevent tensor dimension mismatch crashes in multi-stage pipelines. If Stage 1 produces an RGB tensor $(H, W, 3)$ and Stage 2 is an edge operator (like Canny or Otsu) expecting a 2D Grayscale tensor $(H, W)$, the shim dynamically applies luminance conversion ($Y = 0.299R + 0.587G + 0.114B$) before passing the buffer to Stage 2.

### Q5: How did you fix the infinite rerun / screen flickering loop?
> **Answer:** The flickering was caused by dual-widget desynchronization (`slider` vs `number_input`) and manual in-body `st.rerun()` calls. We migrated to a single high-precision `st.slider` per parameter and hooked them into native Streamlit `on_change` event callbacks. The callback executes and updates the layer state in-memory **before** the script executes, eliminating all ping-pong oscillations and manual `st.rerun()` calls.

### Q6: What is the mathematical formulation of Otsu's thresholding?
> **Answer:** Otsu determines the threshold $t^*$ that maximizes between-class variance $\sigma_B^2(t)$:
> $$\sigma_B^2(t) = \omega_0(t) \omega_1(t) \Big[\mu_0(t) - \mu_1(t)\Big]^2$$
> Where $\omega_0, \omega_1$ are class probabilities of background and foreground, and $\mu_0, \mu_1$ are class mean intensities. Maximizing between-class variance is mathematically equivalent to minimizing intra-class variance $\sigma_W^2(t)$.

### Q7: What is the difference between Histogram Equalization and Histogram Specification?
> **Answer:** Histogram Equalization transforms an image's CDF into a uniform distribution to maximize dynamic range across all intensities. Histogram Specification (Matching) maps the image's CDF to a **custom, predefined target histogram** distribution $p_z(z)$ by computing $z = G^{-1}(T(r))$, allowing precise stylistic or clinical tone matching.

### Q8: What does the parameter $Q$ do in the Contraharmonic Mean Filter?
> **Answer:** In the contraharmonic equation $f(x,y) = \frac{\sum g^{Q+1}}{\sum g^Q}$:
> - When $Q > 0$, the filter eliminates **Pepper noise** (black impulse pixels).
> - When $Q < 0$, the filter eliminates **Salt noise** (white impulse pixels).
> - When $Q = 0$, it simplifies to the standard Arithmetic Mean filter.
> - When $Q = -1$, it simplifies to the Harmonic Mean filter.

### Q9: How does the Unsharp Masking (USM) coring threshold prevent noise amplification?
> **Answer:** It evaluates high-pass residual magnitude $|\Delta| = |f - f_{\text{smooth}}|$. If $|\Delta| < \text{threshold}$, the residual is gated to zero. This ensures sensor noise and subtle film grain remain unamplified while true macroscopic edges receive the full sharpening boost.

### Q10: How is 3D LUT grading implemented?
> **Answer:** By indexing a 3D color cube lattice (typically $17^3$ or $33^3$ RGB entries). For an input pixel $(R,G,B)$, the fractional coordinates locate the bounding cube vertices, and trilinear interpolation interpolates the transformed output color, allowing non-linear cross-channel grading like Hollywood Teal & Orange.

### Q11: What is the difference between Hue and Saturation in the HSI color space?
> **Answer:** Hue ($H \in [0^\circ, 360^\circ]$) represents pure spectral chromaticity (dominant wavelength, e.g. red, yellow, cyan). Saturation ($S \in [0, 1]$) represents the purity or vibrancy of the color (how much white light is mixed with the pure spectral hue).

### Q12: Why is SSIM preferred over MSE or PSNR for perceptual image quality?
> **Answer:** MSE and PSNR calculate pixel-by-pixel numerical variance independently, ignoring spatial correlation and the human visual system (HVS). Two images with identical MSE can look completely different to human observers. SSIM measures degradation in structural information, luminance, and contrast across local windows, closely correlating with human visual perception.

### Q13: What is the mathematical basis of Canny Edge Detection?
> **Answer:** Canny consists of 4 sequential stages:
> 1. 2D Gaussian smoothing to suppress high-frequency noise.
> 2. Directional gradient computation using Sobel operators ($G = \sqrt{G_x^2+G_y^2}$, $\theta = \arctan(G_y/G_x)$).
> 3. Non-Maximum Suppression (NMS) along the gradient direction to thin edges to 1-pixel width.
> 4. Hysteresis Thresholding with two thresholds ($T_{\text{low}}, T_{\text{high}}$) to link edge segments and eliminate spurious noise.

### Q14: What is Shannon Entropy in digital image processing?
> **Answer:** Shannon entropy $H = -\sum p(r_k) \log_2 p(r_k)$ measures the statistical average information content in bits per pixel. An image with an entirely flat histogram has maximum entropy ($8.0\text{ bits/pixel}$), while a solid black or solid white image has zero entropy ($0.0\text{ bits/pixel}$).

### Q15: How does the Tilt-Shift miniature effect trick the human brain?
> **Answer:** In real-world macro photography, focusing on tiny objects close to the lens produces an extremely narrow depth of field with rapid foreground and background defocus. By applying a linear in-focus optical band with quadratic Gaussian defocus falloff to normal cityscapes, the human visual cortex misinterprets the scene as a miniature diorama photographed close-up.

### Q16: How does the Hough Transform detect lines in an image?
> **Answer:** By mapping collinear edge points $(x_i, y_i)$ into sinusoidal curves in polar parameter space $(\rho, \theta)$ using $\rho = x\cos\theta + y\sin\theta$. Curves originating from points on the same line intersect at a common $(\rho^*, \theta^*)$ coordinate in the accumulator matrix. Peak detection identifies the dominant line equations.

### Q17: What are Morphological Opening and Closing, and what are their practical uses?
> **Answer:**
> - **Opening ($A \circ B = (A \ominus B) \oplus B$):** Erosion followed by dilation. It smooths object contours, breaks thin isthmuses, and removes isolated bright spikes smaller than the structuring element.
> - **Closing ($A \bullet B = (A \oplus B) \ominus B$):** Dilation followed by erosion. It fuses narrow breaks, fills thin gulfs, and fills dark holes smaller than the structuring element.

### Q18: What is the difference between White Top-Hat and Black-Hat transforms?
> **Answer:**
> - **White Top-Hat ($A - (A \circ B)$):** Subtracts the opened image from the source, isolating bright details and elements that are smaller than the structuring element against dark backgrounds.
> - **Black-Hat ($(A \bullet B) - A$):** Subtracts the source from the closed image, isolating dark features and dim holes that are smaller than the structuring element against bright backgrounds.

### Q19: What is the von Kries diagonal model of chromatic adaptation?
> **Answer:** It models color adaptation in the human retina by independently scaling the sensitivities of the Long ($L$), Medium ($M$), and Short ($S$) cone photoreceptors:
> $$L' = L \cdot \frac{L_{\text{target}}}{L_{\text{source}}}, \quad M' = M \cdot \frac{M_{\text{target}}}{M_{\text{source}}}, \quad S' = S \cdot \frac{S_{\text{target}}}{S_{\text{source}}}$$
> This enables digital white balance correction when moving between tungsten ($2800\text{K}$) and daylight ($6500\text{K}$) illuminants.

### Q20: What are Lift, Gamma, and Gain in DaVinci Resolve color grading?
> **Answer:**
> - **Lift:** Adjusts dark shadow tones with maximum impact at black ($0$) and minimal impact at white ($1$).
> - **Gain:** Multiplicatively scales highlight tones with maximum impact at white ($1$) pivoting around black ($0$).
> - **Gamma:** Power-law curve modifying midtone exposure without displacing the absolute black or white anchor points.

### Q21: What is the purpose of the difference heatmap in the viewport?
> **Answer:** It computes the absolute spatial error matrix $\|I_{\text{source}} - I_{\text{processed}}\|$ and maps it through the JET pseudo-color colormap. Blue indicates zero difference, yellow indicates moderate alteration, and red indicates significant high-frequency change, allowing instant visual verification of edge loss or noise removal.

### Q22: How does the Harris Corner Detector distinguish between edges and corners?
> **Answer:** By evaluating the eigenvalues $\lambda_1, \lambda_2$ of the second-moment auto-correlation matrix $M$. The corner response measure $R = \det(M) - k \cdot (\text{trace}(M))^2$ distinguishes regions:
> - If $\lambda_1, \lambda_2$ are both large ($R > 0$): **Corner** (intensity changes rapidly in all directions).
> - If one is large and the other is small ($R < 0$): **Edge** (intensity changes in one direction only).
> - If both are near zero ($|R| \approx 0$): **Flat uniform region**.

### Q23: Why do we use frequency domain filtering instead of large spatial convolution kernels?
> **Answer:** By the **Convolution Theorem**, spatial convolution is equivalent to point-wise multiplication in the frequency domain:
> $$\mathcal{F}\{f(x, y) * h(x, y)\} = F(u, v) \cdot H(u, v)$$
> For large kernels (e.g. $k > 15$), spatial convolution requires $\mathcal{O}(N^2 k^2)$ operations, whereas the Fast Fourier Transform (FFT) requires only $\mathcal{O}(N^2 \log N)$ operations, providing dramatic performance speedups.

### Q24: What is the purpose of Pipeline Recipe Serialization?
> **Answer:** It exports the entire active layer stack—including algorithm keys, user-tuned parameter values, and auto-convert flags—into a standard, human-readable JSON file. This guarantees full scientific reproducibility, allowing other researchers to replicate exact multi-stage image processing pipelines with one click.

### Q25: How does the Interactive Split Wipe operate under the hood?
> **Answer:** Given a user wipe percentage $P \in [0, 100]$:
> 1. Resizes and aligns the processed image to match the source raster dimensions.
> 2. Computes the split boundary column $c = \lfloor \frac{P}{100} \cdot W \rfloor$.
> 3. Constructs a composite canvas using NumPy array slicing: `canvas[:, :c] = source[:, :c]` and `canvas[:, c:] = processed[:, c:]`.
> 4. Inlays a $2\text{px}$ high-contrast white divider line at column $c$ for instantaneous before-and-after visual inspection.

---

*This document is maintained as the primary theoretical reference and viva preparation manual for the **Pixel-Forge** computational vision workbench.*
