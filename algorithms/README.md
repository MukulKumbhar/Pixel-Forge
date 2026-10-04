# 🔬 ImageLab Pro — Algorithms & Mathematical Catalog

This directory contains pure NumPy and OpenCV mathematical implementations for all **66 Digital Image Processing and Photoshop Creative Studio algorithms** in ImageLab Pro. Every module is strictly decoupled from UI code and directly implements academic lab experiments (**Exp 01 to Exp 10**) alongside professional digital content creation tools.

---

## 📑 Domain & Lab Experiment Directory

| Domain | File | Primary Topics | Academic Lab / Industry Role |
| :--- | :--- | :--- | :--- |
| **1. Geometry** | [`geometry.py`](file:///d:/Academics/Digital_Image_Processing/Project/ImageLab/algorithms/geometry.py) | Sampling, Discretization, Quantization, Affine Transforms, Canvas Resize, Lens Distortion, Keystone | **Exp 01, Exp 02, Exp 03** + Optical Tools |
| **2. Radiometric** | [`radiometric.py`](file:///d:/Academics/Digital_Image_Processing/Project/ImageLab/algorithms/radiometric.py) | Negative, Log, Gamma, Contrast Stretch, Bit-Planes, Hist Eq/Spec, Brightness/Contrast, Solarize, Posterize | **Exp 04, Exp 05, Exp 06** + Pop-Art |
| **3. Restoration** | [`restoration.py`](file:///d:/Academics/Digital_Image_Processing/Project/ImageLab/algorithms/restoration.py) | Gaussian/S&P/Speckle Noise, Mean, Median, Min/Max, Contraharmonic, Gaussian Blur, Motion Blur, Bilateral, USM | **Exp 07** + Photoshop USM/Bilateral |
| **4. Colorimetry** | [`colorimetry.py`](file:///d:/Academics/Digital_Image_Processing/Project/ImageLab/algorithms/colorimetry.py) | RGB ↔ HSI, HSI Vector Grading, CMYK, Lift/Gamma/Gain, White Balance, 3D LUT, Vibrance, Sepia, Split-Toning | **Exp 08** + DaVinci Grading Suite |
| **5. Segmentation** | [`segmentation.py`](file:///d:/Academics/Digital_Image_Processing/Project/ImageLab/algorithms/segmentation.py) | Sobel, Prewitt, Laplacian, High-Boost, Canny, Otsu, Iterative Mean, Hough Lines, Adaptive Thresh, Hough Circles, Harris | **Exp 09, Exp 10** + Computer Vision |
| **6. Morphology** | [`morphology.py`](file:///d:/Academics/Digital_Image_Processing/Project/ImageLab/algorithms/morphology.py) | Dilation, Erosion, Opening, Closing, Gradient, Top-Hat, Black-Hat, Boundary Extraction, Hit-or-Miss | Academic Unit 6 |
| **7. Creative Studio** | [`artistic.py`](file:///d:/Academics/Digital_Image_Processing/Project/ImageLab/algorithms/artistic.py) | Hand-Drawn Pencil Sketch, Cel-Shaded Cartoonify, Bas-Relief 3D Emboss, Radial Vignette, Tilt-Shift Miniature Blur | Photoshop Creative Suite |

---

## 📐 Detailed Mathematical Reference by Domain

### 1. Geometric & Spatial Foundations (`geometry.py`)
- **Spatial Sampling & Discretization:** Alters spatial grid density ($f_d[x, y] = f(x \cdot \Delta x, y \cdot \Delta y)$) with Nearest, Bilinear, or Bicubic interpolation.
- **Bit-Depth Quantization:** Reduces gray/color levels ($q = \lfloor r / (256 / 2^k) \rfloor \cdot \frac{255}{2^k - 1}$), demonstrating false contouring.
- **Composite Affine Transformation:** Single 3x3 homogeneous matrix combining rotation, scaling, translation, and shearing around the center.
- **Canvas Resize & Letterbox Fitting:** Rescales canvas with aspect-ratio preservation and clean padding.
- **Optical Lens Distortion (Fisheye/Pincushion):** Simulates or corrects radial barrel/pincushion optical distortion ($r_{\text{dist}} = r(1 + k_1 r^2 + k_2 r^4)$).
- **Perspective Keystone Quad Transform:** 4-corner planar homography for straightening angled document scans and architecture.

---

### 2. Radiometric & Contrast Calibration (`radiometric.py`)
- **Photographic Negative:** Inverts brightness values ($s = 255 - r$).
- **Logarithmic Compression:** Expands dark details while compressing highlights ($s = c \cdot \log(1 + r)$).
- **Power-Law / Gamma Correction:** Exponential contrast mapping ($s = c \cdot r^\gamma$).
- **Piecewise Linear Contrast Stretching:** Maps narrow input interval $[r_1, r_2]$ to $[s_1, s_2]$.
- **Bit-Plane Slicing:** Extracts individual binary bit planes (bit 0 LSB to bit 7 MSB).
- **Histogram Equalization:** Spreads histogram peaks across full dynamic range using the CDF ($s_k = (L - 1) \sum p_r(r_j)$).
- **Histogram Specification (Matching):** Transforms image histogram to match a target profile (Gaussian, Bimodal, High-Key, Low-Key).
- **Photoshop Brightness & Contrast:** Linear exposure offset and non-linear contrast expansion ($g = \alpha \cdot f + \beta$).
- **Sabattier Darkroom Solarization:** Pseudo-solarization inverting tones above a threshold ($s = 255 - r$ if $r > T$).
- **Color Posterization:** Quantizes continuous tones into discrete graphic poster bands ($q = \text{round}(r / (256 / N)) \cdot \frac{255}{N - 1}$).

---

### 3. Stochastic Denoising & Image Restoration (`restoration.py`)
- **Noise Degradation Models:** Gaussian (thermal electronic noise), Salt & Pepper (impulse transmission spikes), and Multiplicative Speckle (ultrasound radar noise).
- **Arithmetic Mean Filter:** Local neighborhood box averaging ($f = \frac{1}{mn} \sum g(s, t)$).
- **Geometric Mean Filter:** Multiplicative nth-root smoothing ($f = [\prod g(s, t)]^{\frac{1}{mn}}$) preserving fine detail.
- **Contraharmonic Filter (Order Q):** Non-linear filter ($f = \frac{\sum g^{Q+1}}{\sum g^Q}$) for salt ($Q < 0$) or pepper ($Q > 0$) noise.
- **Order-Statistic Median Filter:** Non-linear median filter for salt-and-pepper removal without edge blurring.
- **Alpha-Trimmed Mean Filter:** Trims extreme $d$ outliers and averages the remaining pixels.
- **Min / Max / Midpoint Filter:** Spatial extrema probing.
- **Gaussian Spatial Smoothing:** Normal distribution bell curve convolution ($G = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$).
- **Directional Motion Blur:** Linear convolution along an angle $\theta$ simulating subject movement or camera shake.
- **Bilateral Filter (Smart Blur / Skin Retouch):** Edge-preserving smoothing combining spatial closeness with photometric radiometric similarity.
- **Photoshop Unsharp Masking (USM):** Gold-standard high-frequency edge definition ($f_{\text{sharp}} = f + \text{amount} \cdot (f - f_{\text{blur}})$).

---

### 4. Chromatic Engine & Color Grading (`colorimetry.py`)
- **Trigonometric RGB ↔ HSI Conversion:** Perceptual color decomposition into Hue, Saturation, and Intensity.
- **HSI Vector Grading:** Shifts Hue angle ($\Delta H$), scales Saturation, and offsets Intensity independently.
- **Three-Way Lift/Gamma/Gain:** Film industry standard grading for dark shadows, neutral midtones, and bright highlights.
- **Von Kries Chromatic Adaptation:** White balance correction using Gray World assumption.
- **CMYK Subtractive Print Decomposition:** Prepress color separation into Cyan, Magenta, Yellow, and Key Black ink plates.
- **Multi-Channel Linear RGB Mixer:** 3x3 channel mixing matrix cross-blending spectral bands.
- **Photoshop Vibrance & Smart Saturation:** Selectively increases saturation of muted colors while preventing oversaturation of skin tones.
- **Antique Sepia Tone Tint:** Nostalgic warm reddish-brown vintage silver photographic print transform.
- **DaVinci Resolve Split Toning:** Complementary shadow (e.g. teal) and highlight (e.g. amber) chromatic split grading.
- **3D Cinematic LUT Color Transform:** Trilinear interpolation through tetrahedral 3D lookup tables (Teal & Orange, Vintage Warm, Bleach Bypass, Cyberpunk).

---

### 5. Feature Segmentation & Structural Analysis (`segmentation.py`)
- **Sobel Operator:** Convolves with 3x3 horizontal and vertical differential kernels ($M = \sqrt{G_x^2 + G_y^2}$).
- **Prewitt Operator:** Uniform 3x3 directional difference filter.
- **Laplacian Second-Derivative Filter:** Second-order spatial derivative ($\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$) highlighting fine zero-crossings.
- **Unsharp Masking & High-Boost Filtering:** Mask amplification ($f_{\text{hb}} = A \cdot f - f_{\text{smooth}}$).
- **Canny Multi-Stage Edge Detector:** Multi-step pipeline with Gaussian smoothing, Sobel gradients, non-maximum suppression, and dual-threshold hysteresis.
- **Global Manual Binarization:** Fixed threshold binary slicing.
- **Otsu Automatic Variance Thresholding:** Maximizes between-class variance ($\sigma_B^2$) across bimodal histograms.
- **Iterative Optimal Mean Thresholding:** Iteratively divides pixel clusters until threshold convergence.
- **Hough Line Transform:** Vector straight line extraction via accumulator voting in $(\rho, \theta)$ space.
- **Local Adaptive Gaussian Binarization:** Local window thresholding ($T(x, y) = \text{Gaussian}(S_{xy}) - C$) for uneven lighting.
- **Hough Circle Transform:** Automated gradient voting in $(a, b, r)$ parameter space for circular objects and pupils.
- **Harris Corner Detection:** Auto-correlation response ($R = \det(M) - k \cdot (\text{trace}(M))^2$) detecting rotation-invariant corners.

---

### 6. Morphological Mechanics (`morphology.py`)
- **Dilation ($A \oplus B$):** Expands foreground object boundaries and bridges small interior holes.
- **Erosion ($A \ominus B$):** Shrinks object boundaries and strips away thin noise spurs.
- **Opening ($A \circ B = (A \ominus B) \oplus B$):** Erosion followed by dilation; eliminates isolated bright noise spikes while preserving shape.
- **Closing ($A \bullet B = (A \oplus B) \ominus B$):** Dilation followed by erosion; fills narrow gaps, bridges breaks, and seals dark holes.
- **Morphological Gradient ($G(A) = (A \oplus B) - (A \ominus B)$):** Isolates perimeter outlines.
- **White Top-Hat Transform ($T_{\text{hat}} = A - (A \circ B)$):** Extracts elements brighter than their surroundings.
- **Black-Hat (Bottom-Hat) Transform ($B_{\text{hat}} = (A \bullet B) - A$):** Isolates dark structures, depressions, and cracks.
- **Boundary Extraction ($\beta(A) = A - (A \ominus B)$):** Subtracts internal eroded core from foreground set.
- **Hit-or-Miss Transform ($A \circledast B$):** Dual structuring element pattern matching for corners, isolated points, and endpoints.

---

### 7. Creative FX & Photoshop Studio (`artistic.py`)
- **Hand-Drawn Pencil Sketch:** Inverts Gaussian blur and blends it with grayscale raster via Color Dodge ($\text{Sketch} = \min(255, \frac{\text{Gray} \cdot 256}{255 - \text{Blur} + 1})$).
- **Cel-Shaded Cartoonify:** Multi-pass bilateral color flattening combined with adaptive ink outline overlays.
- **3D Directional Bas-Relief Emboss:** Convolves image with a directional lighting gradient mask to simulate metallic or stone relief.
- **Radial Optical Lens Vignette:** Quadratic radial falloff shading ($V(r) = 1 - \text{intensity} \cdot (r / R_{\max})^2$).
- **Tilt-Shift Miniature Blur:** Defocus gradient outside a selectable sharp focal band turning real scenes into tabletop dioramas.

---

## 🧪 Verification & Testing

Every single algorithm is asserted for validity, contract compatibility, and shape retention:
```powershell
python tests/test_pipeline.py
```
> Result: **66 passed, 0 failed out of 66 algorithms (100% Pass Rate)**.
