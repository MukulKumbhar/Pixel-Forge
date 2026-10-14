"""
ImageLab Pro - Central Algorithm Registry
Maintains comprehensive metadata for all DIP algorithms:
- Parameter specifications (defaults, bounds, steps, UI types)
- State contracts (expected input format and produced output format)
- Mathematical equations (LaTeX)
- Short definitions and practical real-world examples
"""

from typing import Any, Callable, Dict, List
import algorithms.geometry as geo
import algorithms.radiometric as rad
import algorithms.restoration as rest
import algorithms.colorimetry as col
import algorithms.segmentation as seg
import algorithms.morphology as morph
import algorithms.frequency as freq
import algorithms.artistic as art
import algorithms.compression as comp

# Domain definitions mapping academic syllabi to modern professional roles
DOMAINS = {
    "geometry": {
        "name": "1. Geometric & Spatial Foundations",
        "title": "📐 Size, Position & Shapes (Geometric)",
        "unit": "Unit 1",
        "description": "Spatial discretization, sampling resolutions, bit-depth quantization, affine transformations, canvas geometry, lens distortion, and vector shape cropping."
    },
    "radiometric": {
        "name": "2. Radiometric & Contrast Calibration",
        "title": "☀️ Brightness, Contrast & Tone (Radiometric)",
        "unit": "Unit 2",
        "description": "Point transformations, tone mapping, dynamic range expansion, transfer functions, cumulative distribution processing, and posterization."
    },
    "restoration": {
        "name": "3. Stochastic Denoising & Image Restoration",
        "title": "🧼 Denoise, Blur & Sharpen (Restoration)",
        "unit": "Unit 3",
        "description": "Degradation modeling, inverse filtering, spatial order-statistics, Gaussian/motion blur, bilateral smoothing, and unsharp masking."
    },
    "colorimetry": {
        "name": "4. Chromatic Engine & Color Grading",
        "title": "🎨 Colors & Balance (Colorimetry)",
        "unit": "Unit 4",
        "description": "Color space decomposition (RGB, HSI, CMYK), chromatic adaptation, vector scopes, multi-channel balance, vibrance, sepia, and split-toning."
    },
    "segmentation": {
        "name": "5. Feature Segmentation & Structural Analysis",
        "title": "✂️ Outlines, Edges & Black/White (Segmentation)",
        "unit": "Unit 5",
        "description": "Discontinuity detection, differential edge operators, hysteresis edge linking, parameter-space voting (Hough lines/circles), adaptive thresholding, and Harris corners."
    },
    "morphology": {
        "name": "6. Morphological Mechanics",
        "title": "🧱 Shape Mechanics & Cleaning (Morphology)",
        "unit": "Unit 6",
        "description": "Set-theoretic geometry, topological probes (structuring elements), boundary extraction, top-hat, black-hat, and hit-or-miss transforms."
    },
    "artistic": {
        "name": "7. Creative FX & Photoshop Studio",
        "title": "✨ Artistic & Creative FX (Photoshop Studio)",
        "unit": "Creative Studio",
        "description": "Professional creative filters, Photoshop artistic effects (Pencil Sketch, Cartoonify, Bas-Relief Emboss, Radial Vignette, Tilt-Shift Miniature)."
    },
    "compression": {
        "name": "8. Information Redundancy & Image Compression",
        "title": "🗜️ Compress & Reduce File Size (Compression)",
        "unit": "Unit 8",
        "description": "Coding redundancy, interpixel correlation, psychovisual quantization, 8x8 block Discrete Cosine Transform (DCT) JPEG simulation, and color quantization."
    }
}

ALGORITHM_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Domain 1: Geometric & spatial
    "geo_sampling": {
        "name": "Spatial Sampling & Discretization",
        "friendly_name": "Pixelate / Lower Resolution",
        "simple_description": "Reduces pixel count to create retro pixel art or test downsampling resolution.",
        "tooltip": "Shrink pixel resolution or simulate retro pixelated gaming art.",
        "domain": "geometry",
        "func": geo.spatial_downsample_upsample,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"f_d[x, y] = f(x \cdot \Delta x, y \cdot \Delta y)",
        "summary": "Changes spatial resolution by reducing or increasing pixel grid density.",
        "example": "Generates image thumbnails or simulates pixelation and aliasing effects.",
        "viva_theory": "Changes spatial resolution by reducing or increasing pixel grid density.",
        "params": {
            "scale_factor": {"type": "float", "label": "Sampling Scale", "default": 0.25, "min": 0.05, "max": 1.0, "step": 0.05},
            "interpolation": {"type": "select", "label": "Reconstruction Interpolation", "default": "nearest", "options": ["nearest", "bilinear", "bicubic"]}
        }
    },
    "geo_quantization": {
        "name": "Bit-Depth Quantization",
        "friendly_name": "Retro Bit-Depth (Pixel Art / Posterize)",
        "simple_description": "Reduces color steps from 8-bit down to 1-bit for vintage arcade / retro poster visuals.",
        "tooltip": "Reduce color bit-depth to create vintage arcade / retro computer visuals.",
        "domain": "geometry",
        "func": geo.bit_depth_quantization,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"q = \lfloor r / (256 / 2^k) \rfloor \cdot (255 / (2^k - 1))",
        "summary": "Reduces the number of bits representing gray or color levels (e.g. from 8-bit to 1-bit).",
        "example": "Demonstrates false contouring (banding) and retro low-bit posterization.",
        "viva_theory": "Reduces the number of bits representing gray or color levels (e.g. from 8-bit to 1-bit).",
        "params": {
            "bits": {"type": "int", "label": "Quantization Bit-Depth (k)", "default": 3, "min": 1, "max": 7, "step": 1}
        }
    },
    "geo_affine": {
        "name": "Composite Affine Transformation",
        "friendly_name": "Rotate, Scale, Shift & Tilt",
        "simple_description": "Rotate canvas by degrees, zoom in/out, shift position, or apply 2D slant skew.",
        "tooltip": "Rotate, scale, translate, or shear the canvas in 2D space.",
        "domain": "geometry",
        "func": geo.affine_transform,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"\mathbf{x}' = \mathbf{T}_{center} \cdot \mathbf{Sh} \cdot \mathbf{S} \cdot \mathbf{R}(\theta) \cdot \mathbf{T}_{center}^{-1} \cdot \mathbf{x}",
        "summary": "Applies 2D rotation, scaling, translation, and shearing using matrix multiplication.",
        "example": "Corrects tilted document scans and rotates camera photos.",
        "viva_theory": "Applies 2D rotation, scaling, translation, and shearing using matrix multiplication.",
        "params": {
            "angle": {"type": "float", "label": "Rotation Angle (°)", "default": 15.0, "min": -180.0, "max": 180.0, "step": 1.0},
            "scale_x": {"type": "float", "label": "Scale X", "default": 1.0, "min": 0.2, "max": 2.5, "step": 0.05},
            "scale_y": {"type": "float", "label": "Scale Y", "default": 1.0, "min": 0.2, "max": 2.5, "step": 0.05},
            "trans_x": {"type": "float", "label": "Translate X (px)", "default": 0.0, "min": -150.0, "max": 150.0, "step": 5.0},
            "trans_y": {"type": "float", "label": "Translate Y (px)", "default": 0.0, "min": -150.0, "max": 150.0, "step": 5.0},
            "shear_x": {"type": "float", "label": "Shear X", "default": 0.0, "min": -0.8, "max": 0.8, "step": 0.05},
            "shear_y": {"type": "float", "label": "Shear Y", "default": 0.0, "min": -0.8, "max": 0.8, "step": 0.05}
        }
    },
    "geo_canvas_resize": {
        "name": "Canvas Resize & Letterbox Fitting",
        "friendly_name": "Resize Canvas & Letterbox",
        "simple_description": "Changes image resolution with clean aspect-ratio preservation and letterbox borders.",
        "tooltip": "Resize canvas to custom pixel dimensions with optional letterboxing.",
        "domain": "geometry",
        "func": geo.canvas_resize,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"I_{new}(x', y') = I(x, y) \quad (\text{with aspect padding})",
        "summary": "Resizes image canvas with aspect-ratio preservation and clean letterbox padding.",
        "example": "Prepares images for social media platforms or fixed-size deep learning inputs.",
        "viva_theory": "Resizes image canvas with aspect-ratio preservation and clean letterbox padding.",
        "params": {
            "width": {"type": "int", "label": "Canvas Target Width", "default": 512, "min": 128, "max": 1920, "step": 64},
            "height": {"type": "int", "label": "Canvas Target Height", "default": 512, "min": 128, "max": 1080, "step": 64},
            "mode": {"type": "select", "label": "Scaling Mode", "default": "fit", "options": ["fit", "stretch"]}
        }
    },
    "geo_lens_distortion": {
        "name": "Optical Lens Distortion (Fisheye/Pincushion)",
        "friendly_name": "Fisheye & Action-Cam Warp",
        "simple_description": "Simulates curved GoPro fisheye action-cam bubble or corrects barrel distortion.",
        "tooltip": "Apply curved fisheye bubble or correct wide-angle lens distortion.",
        "domain": "geometry",
        "func": geo.lens_distortion,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"r_{distorted} = r \cdot (1 + k_1 r^2 + k_2 r^4)",
        "summary": "Simulates or corrects radial barrel and pincushion optical camera lens distortion.",
        "example": "Creates wide-angle GoPro action cam fisheye look or straightens curved architectural horizons.",
        "viva_theory": "Simulates or corrects radial barrel and pincushion optical camera lens distortion.",
        "params": {
            "k1": {"type": "float", "label": "Radial Coefficient (k1)", "default": -0.20, "min": -0.60, "max": 0.60, "step": 0.02},
            "k2": {"type": "float", "label": "Higher Order (k2)", "default": 0.0, "min": -0.20, "max": 0.20, "step": 0.01}
        }
    },
    "geo_perspective": {
        "name": "Perspective Keystone Quad Transform",
        "friendly_name": "Straighten Angle / Perspective",
        "simple_description": "Fixes tilted angles or keystone distortion from angled photos and document scans.",
        "tooltip": "Correct keystoning or simulate tilted viewpoint angles.",
        "domain": "geometry",
        "func": geo.perspective_keystone,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"\mathbf{x}' = \mathbf{H} \cdot \mathbf{x}",
        "summary": "Warps four corner coordinates to simulate perspective tilt or straighten angled document scans.",
        "example": "Straightens tall buildings photographed from the ground or scans taken at an angle.",
        "viva_theory": "Warps four corner coordinates to simulate perspective tilt or straighten angled document scans.",
        "params": {
            "tilt_x": {"type": "float", "label": "Horizontal Keystone Tilt", "default": 0.10, "min": -0.40, "max": 0.40, "step": 0.02},
            "tilt_y": {"type": "float", "label": "Vertical Keystone Tilt", "default": 0.0, "min": -0.40, "max": 0.40, "step": 0.02}
        }
    },
    "geo_shape_crop": {
        "name": "Vector Multi-Shape Cropper",
        "friendly_name": "Cut into Shape (Circle, Square, Star, Heart)",
        "simple_description": "Crops image into 11 geometric vector shapes with smooth feathered edges and alpha transparency.",
        "tooltip": "Crop canvas into circle, square, star, heart, or polygon with smooth alpha transparency.",
        "domain": "geometry",
        "func": geo.vector_shape_crop,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"I_{crop}(x, y) = I(x, y) \cdot M_{shape}(x, y)",
        "summary": "Crops canvas using analytical geometric shapes (Circle, Square, Rectangle, Ellipse, Triangle, Diamond, Hexagon, Octagon, Star, Heart, Polygon) with anti-aliasing and transparency.",
        "example": "Creates circular avatar profile photos, 1:1 square crops for Instagram, or custom star and heart badge stickers.",
        "viva_theory": "Analytical vector rasterization utilizing subpixel polygon boundary integration, edge anti-aliasing, Gaussian feathering, and alpha-compositing.",
        "params": {
            "shape": {"type": "select", "label": "Crop Geometry Shape", "default": "circle", "options": ["circle", "square", "rectangle", "ellipse", "triangle", "diamond", "hexagon", "octagon", "star", "heart", "polygon"]},
            "center_x": {"type": "float", "label": "Center X Position", "default": 0.50, "min": 0.0, "max": 1.0, "step": 0.02},
            "center_y": {"type": "float", "label": "Center Y Position", "default": 0.50, "min": 0.0, "max": 1.0, "step": 0.02},
            "size": {"type": "float", "label": "Shape Radius / Scale", "default": 0.80, "min": 0.10, "max": 1.50, "step": 0.05},
            "aspect_ratio": {"type": "select", "label": "Aspect Ratio Preset", "default": "1:1", "options": ["1:1", "4:3", "16:9", "3:2", "9:16", "freeform"]},
            "rotation": {"type": "float", "label": "Rotation Angle (°)", "default": 0.0, "min": -180.0, "max": 180.0, "step": 5.0},
            "feather": {"type": "int", "label": "Edge Feathering Softness (px)", "default": 0, "min": 0, "max": 50, "step": 1},
            "bg_fill": {"type": "select", "label": "Background Fill Mode", "default": "transparent", "options": ["transparent", "black", "white", "studio_dark"]},
            "tight_crop": {"type": "bool", "label": "Crop Tight to Bounding Box", "default": True},
            "sides": {"type": "int", "label": "Polygon Sides (if N-gon)", "default": 5, "min": 3, "max": 16, "step": 1},
            "star_inner_ratio": {"type": "float", "label": "Star Inner Radius Ratio", "default": 0.382, "min": 0.15, "max": 0.85, "step": 0.05}
        }
    },

    # Domain 2: Radiometric & contrast
    "rad_negative": {
        "name": "Digital Photographic Negative",
        "friendly_name": "Invert Colors / Photo Negative",
        "simple_description": "Flips bright into dark and dark into bright, just like a classic film negative.",
        "tooltip": "Flip all image colors into their photographic negative.",
        "domain": "radiometric",
        "func": rad.image_negative,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"s = L - 1 - r",
        "summary": "Inverts pixel intensity values so bright areas become dark and dark areas become bright.",
        "example": "Enhances white or grey details embedded in large dark regions, such as medical mammograms.",
        "viva_theory": "Inverts pixel intensity values so bright areas become dark and dark areas become bright.",
        "params": {}
    },
    "rad_log": {
        "name": "Logarithmic Dynamic Range Compression",
        "friendly_name": "Brighten Dark Shadows (Log)",
        "simple_description": "Reveals hidden details in deep dark shadows without overexposing bright highlights.",
        "tooltip": "Expand dark shadow details while compressing bright highlights.",
        "domain": "radiometric",
        "func": rad.log_transform,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"s = c \cdot \log(1 + r)",
        "summary": "Expands dark pixel values while compressing higher bright pixel values.",
        "example": "Visualizes Fourier transform magnitude spectra where central DC peaks dominate.",
        "viva_theory": "Expands dark pixel values while compressing higher bright pixel values.",
        "params": {
            "c": {"type": "float", "label": "Scaling Constant (c)", "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}
        }
    },
    "rad_gamma": {
        "name": "Power-Law (Gamma) Correction",
        "friendly_name": "Adjust Midtone Lightness (Gamma)",
        "simple_description": "Lightens or darkens midtones smoothly while keeping true blacks and whites untouched.",
        "tooltip": "Tune midtone brightness without shifting black or white points.",
        "domain": "radiometric",
        "func": rad.power_law_gamma,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"s = c \cdot r^\gamma",
        "summary": "Maps input gray levels to output levels using an exponential power curve.",
        "example": "Calibrates display monitor color reproduction and fixes underexposed shadows (gamma < 1).",
        "viva_theory": "Maps input gray levels to output levels using an exponential power curve.",
        "params": {
            "gamma": {"type": "float", "label": "Gamma Exponent (\u03b3)", "default": 0.5, "min": 0.1, "max": 3.0, "step": 0.05},
            "c": {"type": "float", "label": "Scaling Constant (c)", "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}
        }
    },
    "rad_contrast_stretch": {
        "name": "Piecewise-Linear Contrast Stretching",
        "friendly_name": "Highlight & Shadow Stretcher",
        "simple_description": "Expands washed-out, narrow gray ranges to make deep blacks and punchy brights.",
        "tooltip": "Stretch narrow contrast ranges to make faded images punchy and crisp.",
        "domain": "radiometric",
        "func": rad.contrast_stretch_piecewise,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"s = \frac{s_2 - s_1}{r_2 - r_1}(r - r_1) + s_1",
        "summary": "Expands a narrow range of brightness levels into the full 0-255 dynamic range.",
        "example": "Improves clarity and details in low-contrast washed-out or foggy photos.",
        "viva_theory": "Expands a narrow range of brightness levels into the full 0-255 dynamic range.",
        "params": {
            "r1": {"type": "int", "label": "Input Low Breakpoint (r1)", "default": 50, "min": 0, "max": 127, "step": 1},
            "s1": {"type": "int", "label": "Output Low Target (s1)", "default": 10, "min": 0, "max": 127, "step": 1},
            "r2": {"type": "int", "label": "Input High Breakpoint (r2)", "default": 200, "min": 128, "max": 255, "step": 1},
            "s2": {"type": "int", "label": "Output High Target (s2)", "default": 245, "min": 128, "max": 255, "step": 1}
        }
    },
    "rad_bit_plane": {
        "name": "Bit-Plane Slicing",
        "friendly_name": "Make Bit-Wise Image (Bit Planes 1-8)",
        "simple_description": "Visualizes individual digital bit layers from subtle noise patterns up to main outlines.",
        "tooltip": "Inspect specific binary bit-planes from LSB noise to MSB structural outlines.",
        "domain": "radiometric",
        "func": rad.bit_plane_slice,
        "input_contract": "ANY",
        "output_contract": "BINARY",
        "formula": r"b_k(x, y) = \text{bitand}(f(x, y), 2^k)",
        "summary": "Extracts the contribution made by individual bits (from Bit 0 LSB to Bit 7 MSB).",
        "example": "Used in digital watermarking (hiding data in LSB) and image compression.",
        "viva_theory": "Extracts the contribution made by individual bits (from Bit 0 LSB to Bit 7 MSB).",
        "params": {
            "plane": {"type": "int", "label": "Bit Plane (0=LSB, 7=MSB)", "default": 7, "min": 0, "max": 7, "step": 1}
        }
    },
    "rad_hist_eq": {
        "name": "Histogram Equalization",
        "friendly_name": "Auto-Enhance Contrast (Histogram Equalize)",
        "simple_description": "Automatically redistributes lighting tones across the full spectrum for instant maximum clarity.",
        "tooltip": "Spread tonal brightness across dynamic range automatically.",
        "domain": "radiometric",
        "func": rad.histogram_equalization,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"s_k = T(r_k) = (L - 1) \sum_{j=0}^k p_r(r_j)",
        "summary": "Flattens and spreads out the intensity histogram using the Cumulative Distribution Function (CDF).",
        "example": "Automatically reveals hidden details in medical X-rays and dark night surveillance footage.",
        "viva_theory": "Flattens and spreads out the intensity histogram using the Cumulative Distribution Function (CDF).",
        "params": {}
    },
    "rad_hist_spec": {
        "name": "Histogram Specification (Matching)",
        "friendly_name": "Match Lighting Style (Histogram Match)",
        "simple_description": "Reshapes image brightness distribution to match a chosen target lighting style.",
        "tooltip": "Reshape image tones to match a desired lighting distribution profile.",
        "domain": "radiometric",
        "func": rad.histogram_specification,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"z_k = G^{-1}(T(r_k))",
        "summary": "Transforms the image histogram so that it matches a specified target distribution shape.",
        "example": "Matches the lighting and tonal mood across satellite images taken at different times of day.",
        "viva_theory": "Transforms the image histogram so that it matches a specified target distribution shape.",
        "params": {
            "target_profile": {"type": "select", "label": "Target Distribution Profile", "default": "gaussian", "options": ["gaussian", "bimodal", "uniform", "high_key", "low_key"]}
        }
    },
    "rad_brightness_contrast": {
        "name": "Photoshop Brightness & Contrast",
        "friendly_name": "Brightness & Contrast",
        "simple_description": "Classic photo controls to make the picture lighter, darker, punchier, or softer.",
        "tooltip": "Make the image lighter/darker and increase or decrease contrast punch.",
        "domain": "radiometric",
        "func": rad.brightness_contrast,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"g(x, y) = \alpha \cdot f(x, y) + \beta",
        "summary": "Standard linear brightness offset and non-linear contrast expansion.",
        "example": "Quickly balances overall exposure and deepens blacks in washed-out photographs.",
        "viva_theory": "Standard linear brightness offset and non-linear contrast expansion.",
        "params": {
            "brightness": {"type": "int", "label": "Brightness Offset", "default": 15, "min": -100, "max": 100, "step": 1},
            "contrast": {"type": "int", "label": "Contrast Gain", "default": 20, "min": -100, "max": 100, "step": 1}
        }
    },
    "rad_solarize": {
        "name": "Sabattier Darkroom Solarization",
        "friendly_name": "Darkroom Solarize (Sabatier Glow)",
        "simple_description": "Inverts highlight tones above a cutoff to create surreal glowing edge halos.",
        "tooltip": "Invert highlight tones above a cutoff to create surreal glowing edges.",
        "domain": "radiometric",
        "func": rad.solarize_effect,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"s = 255 - r \quad (\text{if } r > T)",
        "summary": "Inverts tone values exceeding a threshold to produce physical darkroom pseudo-solarization.",
        "example": "Creates psychedelic surreal halo effects around high-contrast portrait highlights.",
        "viva_theory": "Inverts tone values exceeding a threshold to produce physical darkroom pseudo-solarization.",
        "params": {
            "threshold": {"type": "int", "label": "Solarization Cutoff", "default": 128, "min": 32, "max": 224, "step": 4}
        }
    },
    "rad_posterize": {
        "name": "Color Posterization Tonal Bands",
        "friendly_name": "Posterize (Color Bands)",
        "simple_description": "Reduces continuous color gradients into bold flat graphic art poster bands.",
        "tooltip": "Reduce tonal levels into distinct flat graphic poster bands.",
        "domain": "radiometric",
        "func": rad.posterize_levels,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"q = \text{round}(r / (256 / N)) \cdot (255 / (N - 1))",
        "summary": "Reduces continuous color gradations into discrete graphic poster bands.",
        "example": "Transforms photographs into stylized pop-art screenprints.",
        "viva_theory": "Reduces continuous color gradations into discrete graphic poster bands.",
        "params": {
            "levels": {"type": "int", "label": "Tonal Steps (N)", "default": 4, "min": 2, "max": 16, "step": 1}
        }
    },

    # Domain 3: Restoration & filtering
    "rest_noise_gaussian": {
        "name": "Additive Gaussian Noise Injection",
        "friendly_name": "Add Film Grain / Camera Noise",
        "simple_description": "Injects subtle electronic sensor noise or vintage analogue photographic film grain.",
        "tooltip": "Inject realistic sensor noise or vintage film grain.",
        "domain": "restoration",
        "func": rest.add_gaussian_noise,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"p(z) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(z - \mu)^2}{2\sigma^2}}",
        "summary": "Adds electronic sensor noise following a normal bell-curve probability distribution.",
        "example": "Simulates poor lighting sensor noise in digital cameras.",
        "viva_theory": "Adds electronic sensor noise following a normal bell-curve probability distribution.",
        "params": {
            "mean": {"type": "float", "label": "Noise Mean (\u03bc)", "default": 0.0, "min": -50.0, "max": 50.0, "step": 1.0},
            "sigma": {"type": "float", "label": "Standard Deviation (\u03c3)", "default": 25.0, "min": 1.0, "max": 100.0, "step": 1.0}
        }
    },
    "rest_noise_sp": {
        "name": "Salt & Pepper (Impulse) Noise",
        "friendly_name": "Add Speckles (Salt & Pepper Noise)",
        "simple_description": "Simulates sensor dust or wireless transmission errors with random white and black dots.",
        "tooltip": "Add random white and black impulse noise speckles.",
        "domain": "restoration",
        "func": rest.add_salt_and_pepper_noise,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"P(z) = \begin{cases} P_a & \text{for } z = 0 \text{ (pepper)} \\ P_b & \text{for } z = 255 \text{ (salt)} \end{cases}",
        "summary": "Randomly corrupts pixels by setting them to pure white (salt) or pure black (pepper).",
        "example": "Simulates faulty camera sensors or transmission bit errors over wireless channels.",
        "viva_theory": "Randomly corrupts pixels by setting them to pure white (salt) or pure black (pepper).",
        "params": {
            "salt_prob": {"type": "float", "label": "Salt Probability (Pa)", "default": 0.02, "min": 0.0, "max": 0.2, "step": 0.01},
            "pepper_prob": {"type": "float", "label": "Pepper Probability (Pb)", "default": 0.02, "min": 0.0, "max": 0.2, "step": 0.01}
        }
    },
    "rest_noise_speckle": {
        "name": "Multiplicative Speckle Noise",
        "friendly_name": "Add Multiplicative Speckle Noise",
        "simple_description": "Adds textured grain that scales with brightness, simulating ultrasound scans and radar.",
        "tooltip": "Simulate intensity-dependent radar or ultrasound speckle grain.",
        "domain": "restoration",
        "func": rest.add_speckle_noise,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"g(x, y) = f(x, y) + f(x, y) \cdot n(x, y)",
        "summary": "Multiplies pixel values by random noise, where corruption scales with local intensity.",
        "example": "Simulates granular interference seen in medical Ultrasound and radar (SAR) imagery.",
        "viva_theory": "Multiplies pixel values by random noise, where corruption scales with local intensity.",
        "params": {
            "variance": {"type": "float", "label": "Noise Variance (\u03c3\u00b2)", "default": 0.05, "min": 0.01, "max": 0.5, "step": 0.01}
        }
    },
    "rest_arithmetic_mean": {
        "name": "Arithmetic Mean Spatial Filter",
        "friendly_name": "Soft Average Blur (Box Filter)",
        "simple_description": "Evenly averages neighboring pixels to soften grain, noise, and sharp textures.",
        "tooltip": "Uniformly smooth and blur neighborhood pixels.",
        "domain": "restoration",
        "func": rest.arithmetic_mean_filter,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"\hat{f}(x, y) = \frac{1}{mn} \sum_{(s,t) \in S_{xy}} g(s, t)",
        "summary": "Replaces each pixel with the average value of its local neighborhood.",
        "example": "Smoothes out uniform random background noise, but blurs sharp object boundaries.",
        "viva_theory": "Replaces each pixel with the average value of its local neighborhood.",
        "params": {
            "ksize": {"type": "int", "label": "Kernel Dimension (k)", "default": 3, "min": 3, "max": 15, "step": 2}
        }
    },
    "rest_geometric_mean": {
        "name": "Geometric Mean Spatial Filter",
        "friendly_name": "Smooth Detail Blur (Geometric Mean)",
        "simple_description": "Softens noise gently while preserving subtle textures better than simple averaging.",
        "tooltip": "Gentle smoothing with better edge retention than box averaging.",
        "domain": "restoration",
        "func": rest.geometric_mean_filter,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"\hat{f}(x, y) = \left[ \prod_{(s,t) \in S_{xy}} g(s, t) \right]^{\frac{1}{mn}}",
        "summary": "Multiplies neighborhood pixels together and takes the mn-th root.",
        "example": "Achieves smoothing comparable to arithmetic mean while preserving finer image details.",
        "viva_theory": "Multiplies neighborhood pixels together and takes the mn-th root.",
        "params": {
            "ksize": {"type": "int", "label": "Kernel Dimension (k)", "default": 3, "min": 3, "max": 11, "step": 2}
        }
    },
    "rest_contraharmonic": {
        "name": "Contraharmonic Mean Filter",
        "friendly_name": "Clean Speckle Noise (Contraharmonic)",
        "simple_description": "Specialized filter designed to selectively wipe out either dark spots or white specks.",
        "tooltip": "Erase isolated dark or white impulse noise spots.",
        "domain": "restoration",
        "func": rest.contraharmonic_mean_filter,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"\hat{f}(x, y) = \frac{\sum g(s, t)^{Q+1}}{\sum g(s, t)^Q}",
        "summary": "Specialized non-linear filter suited for eliminating salt or pepper noise based on order Q.",
        "example": "Positive Q eliminates pepper noise; negative Q eliminates salt noise.",
        "viva_theory": "Specialized non-linear filter suited for eliminating salt or pepper noise based on order Q.",
        "params": {
            "ksize": {"type": "int", "label": "Kernel Dimension (k)", "default": 3, "min": 3, "max": 9, "step": 2},
            "Q": {"type": "float", "label": "Filter Order (Q)", "default": 1.5, "min": -3.0, "max": 3.0, "step": 0.5}
        }
    },
    "rest_median": {
        "name": "Order-Statistic Median Filter",
        "friendly_name": "Remove Dust & Speckles (Median Denoise)",
        "simple_description": "The gold standard for removing salt-and-pepper dust and scratches without blurring sharp edges.",
        "tooltip": "Wipe out salt and pepper noise while keeping object edges razor sharp.",
        "domain": "restoration",
        "func": rest.median_filter,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"\hat{f}(x, y) = \text{median}_{(s,t) \in S_{xy}} \{ g(s, t) \}",
        "summary": "Sorts neighboring pixels and replaces the center pixel with the middle (median) value.",
        "example": "The gold-standard for removing salt & pepper noise without blurring sharp edge outlines.",
        "viva_theory": "Sorts neighboring pixels and replaces the center pixel with the middle (median) value.",
        "params": {
            "ksize": {"type": "int", "label": "Kernel Dimension (k)", "default": 3, "min": 3, "max": 15, "step": 2}
        }
    },
    "rest_alpha_trimmed": {
        "name": "Alpha-Trimmed Mean Filter",
        "friendly_name": "Clean Heavy Mixed Noise (Alpha-Trimmed)",
        "simple_description": "Removes extreme grain and random spikes together by ignoring outlier pixels.",
        "tooltip": "Denoise images corrupted by both grain and impulse speckles.",
        "domain": "restoration",
        "func": rest.alpha_trimmed_mean_filter,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"\hat{f}(x, y) = \frac{1}{mn - d} \sum_{i=d/2}^{mn - d/2} g_{(i)}(s, t)",
        "summary": "Discards the highest and lowest pixel values in the neighborhood and averages the rest.",
        "example": "Filters images containing both Gaussian noise and salt & pepper impulse spikes simultaneously.",
        "viva_theory": "Discards the highest and lowest pixel values in the neighborhood and averages the rest.",
        "params": {
            "ksize": {"type": "int", "label": "Kernel Size", "default": 3, "min": 3, "max": 7, "step": 2},
            "d": {"type": "int", "label": "Trimmed Pixels (d)", "default": 2, "min": 0, "max": 6, "step": 2}
        }
    },
    "rest_order_stats": {
        "name": "Min / Max / Midpoint Filter",
        "friendly_name": "Shrink or Grow Bright Spots (Min/Max/Mid)",
        "simple_description": "Shrinks bright spots, expands dark spots, or finds the middle intensity.",
        "tooltip": "Apply local minimum, maximum, or midpoint order statistics.",
        "domain": "restoration",
        "func": rest.order_statistic_min_max_filter,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"\hat{f}_{\text{mid}}(x, y) = \frac{1}{2} \left[ \max g(s, t) + \min g(s, t) \right]",
        "summary": "Calculates the minimum, maximum, or midpoint pixel value in the local neighborhood.",
        "example": "Min filter removes salt noise; Max filter removes dark pepper spots.",
        "viva_theory": "Calculates the minimum, maximum, or midpoint pixel value in the local neighborhood.",
        "params": {
            "ksize": {"type": "int", "label": "Neighborhood Size", "default": 3, "min": 3, "max": 9, "step": 2},
            "mode": {"type": "select", "label": "Statistic Mode", "default": "midpoint", "options": ["min", "max", "midpoint"]}
        }
    },
    "freq_spectrum": {
        "name": "2D Fourier Magnitude Spectrum",
        "friendly_name": "View Frequency Waves (2D Fourier FFT)",
        "simple_description": "Visualizes your image as frequency waves to spot repeating patterns, grids, or noise bars.",
        "tooltip": "Convert image to 2D Fourier frequency domain magnitude spectrum.",
        "domain": "restoration",
        "func": freq.compute_fft_spectrum,
        "input_contract": "ANY",
        "output_contract": "GRAYSCALE",
        "formula": r"M(u, v) = c \cdot \log(1 + |F(u, v)|), \quad F(u, v) = \mathcal{F}_{2D}\{f(x, y)\}",
        "summary": "Computes the 2D Fast Fourier Transform (FFT) and visualizes frequency distribution.",
        "example": "Identifies periodic noise bars, grid interference, and dominant edge angles.",
        "viva_theory": "Computes the 2D Fast Fourier Transform (FFT) and visualizes frequency distribution.",
        "params": {}
    },
    "freq_filter": {
        "name": "2D Frequency Domain Filtering (FFT)",
        "friendly_name": "Frequency Filter (Low-Pass Blur / High-Pass Detail)",
        "simple_description": "Filters frequencies directly: smooth out waves (Low-Pass) or isolate fine ripples (High-Pass).",
        "tooltip": "Apply ideal, Butterworth, or Gaussian frequency filters via 2D FFT.",
        "domain": "restoration",
        "func": freq.frequency_domain_filter,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"G(u, v) = H(u, v) \cdot F(u, v)",
        "summary": "Filters frequencies in Fourier space using Lowpass or Highpass transfer functions.",
        "example": "Lowpass blurs and denoises; Highpass extracts sharp structural edges.",
        "viva_theory": "Filters frequencies in Fourier space using Lowpass or Highpass transfer functions.",
        "params": {
            "filter_type": {"type": "select", "label": "Filter Transfer Function", "default": "gaussian", "options": ["gaussian", "butterworth", "ideal"]},
            "pass_type": {"type": "select", "label": "Band Response", "default": "lowpass", "options": ["lowpass", "highpass"]},
            "cutoff_d0": {"type": "float", "label": "Cutoff Radius (D0)", "default": 35.0, "min": 5.0, "max": 150.0, "step": 5.0},
            "order_n": {"type": "int", "label": "Butterworth Order (n)", "default": 2, "min": 1, "max": 6, "step": 1}
        }
    },
    "rest_gaussian_blur": {
        "name": "Gaussian Spatial Smoothing",
        "friendly_name": "Smooth Gaussian Blur",
        "simple_description": "The classic natural, silky blur used in photo editing to defocus backgrounds or clean grain.",
        "tooltip": "Apply natural, smooth bell-curve Gaussian blurring.",
        "domain": "restoration",
        "func": rest.gaussian_blur,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}",
        "summary": "Convolves image with a smooth Gaussian bell curve to attenuate high-frequency noise.",
        "example": "Smooths digital sensor noise and prepares images for edge detection.",
        "viva_theory": "Convolves image with a smooth Gaussian bell curve to attenuate high-frequency noise.",
        "params": {
            "ksize": {"type": "int", "label": "Kernel Dimension (k)", "default": 7, "min": 3, "max": 25, "step": 2},
            "sigma": {"type": "float", "label": "Gaussian Sigma (\u03c3)", "default": 1.5, "min": 0.2, "max": 10.0, "step": 0.1}
        }
    },
    "rest_motion_blur": {
        "name": "Directional Motion Blur",
        "friendly_name": "Action Motion Blur (Speed Streak)",
        "simple_description": "Adds a directional speed streak effect in any angle to make subjects look like they are zooming.",
        "tooltip": "Simulate directional camera movement or high-speed subject motion.",
        "domain": "restoration",
        "func": rest.motion_blur,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"h(x, y) = \frac{1}{L} \quad (\text{along angle } \theta)",
        "summary": "Simulates linear camera shake or rapid subject motion along a specified angle.",
        "example": "Adds dynamic speed streaks to racing cars or running sports photography.",
        "viva_theory": "Simulates linear camera shake or rapid subject motion along a specified angle.",
        "params": {
            "size": {"type": "int", "label": "Blur Stroke Length", "default": 15, "min": 3, "max": 45, "step": 2},
            "angle": {"type": "float", "label": "Motion Angle (°)", "default": 45.0, "min": -180.0, "max": 180.0, "step": 5.0}
        }
    },
    "rest_bilateral": {
        "name": "Bilateral Filter (Smart Blur / Skin Retouch)",
        "friendly_name": "Smooth Skin & Retouch (Smart Bilateral)",
        "simple_description": "Retouches skin textures to look smooth and flawless while keeping eyes, lips, and hair sharp.",
        "tooltip": "Smooth facial textures and noise while preserving sharp structural edges.",
        "domain": "restoration",
        "func": rest.bilateral_filter,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"I^{filtered}(x) = \frac{1}{W_p} \sum I(x_i) f_r(\|I(x_i) - I(x)\|) g_s(\|x_i - x\|)",
        "summary": "Non-linear edge-preserving filter that smooths textures while keeping sharp borders crisp.",
        "example": "Industry standard Photoshop tool for digital portrait skin smoothing and de-noising.",
        "viva_theory": "Non-linear edge-preserving filter that smooths textures while keeping sharp borders crisp.",
        "params": {
            "d": {"type": "int", "label": "Pixel Neighborhood Diameter", "default": 9, "min": 3, "max": 15, "step": 2},
            "sigma_color": {"type": "float", "label": "Color Space Sigma", "default": 75.0, "min": 10.0, "max": 150.0, "step": 5.0},
            "sigma_space": {"type": "float", "label": "Coordinate Space Sigma", "default": 75.0, "min": 10.0, "max": 150.0, "step": 5.0}
        }
    },
    "rest_unsharp_mask": {
        "name": "Photoshop Unsharp Masking (USM)",
        "friendly_name": "Sharpen Details (Unsharp Mask)",
        "simple_description": "Crisps up soft edges, fine hairs, and text to make your photo look sharp and high-definition.",
        "tooltip": "Increase crispness and micro-contrast on textures and edges.",
        "domain": "restoration",
        "func": rest.unsharp_mask,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"f_{sharp} = f + \text{amount} \cdot (f - f_{blur})",
        "summary": "The gold-standard sharpening filter in Adobe Photoshop for crisp high-frequency definition.",
        "example": "Restores crisp texture in product catalog photos and scanned documents.",
        "viva_theory": "The gold-standard sharpening filter in Adobe Photoshop for crisp high-frequency definition.",
        "params": {
            "amount": {"type": "float", "label": "Sharpening Strength", "default": 1.5, "min": 0.2, "max": 4.0, "step": 0.1},
            "radius": {"type": "int", "label": "Blur Radius", "default": 3, "min": 3, "max": 15, "step": 2},
            "threshold": {"type": "int", "label": "Noise Threshold Cutoff", "default": 0, "min": 0, "max": 30, "step": 1}
        }
    },

    # Domain 4: Colorimetry & grading
    "col_3d_lut": {
        "name": "3D Cinematic LUT Color Transform",
        "friendly_name": "Cinematic Film Looks (3D LUTs)",
        "simple_description": "Applies blockbuster movie color grading like Hollywood Teal & Orange, Vintage Warm, or Cyberpunk.",
        "tooltip": "Transform colors using professional 3D film lookup tables.",
        "domain": "colorimetry",
        "func": col.apply_3d_lut,
        "input_contract": "RGB",
        "output_contract": "RGB",
        "formula": r"\mathbf{C}_{out} = \text{TrilinearInterp}(\mathbf{LUT}_{3D}, \mathbf{C}_{in})",
        "summary": "Applies a 3D color lookup table for professional cinematic tone and hue remapping.",
        "example": "Transforms standard digital video into warm golden-hour or cool film looks.",
        "viva_theory": "Applies a 3D color lookup table for professional cinematic tone and hue remapping.",
        "params": {
            "profile": {"type": "select", "label": "Film Stock LUT Profile", "default": "teal_orange", "options": ["teal_orange", "vintage_warm", "bleach_bypass", "cyberpunk"]},
            "strength": {"type": "float", "label": "Blend Strength", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}
        }
    },
    "col_hsi_vector": {
        "name": "HSI Geometric Color Vector Grading",
        "friendly_name": "Control HSB (Hue, Saturation, Brightness)",
        "simple_description": "Rotate colors across the rainbow, boost color richness, or adjust lighting smoothly.",
        "tooltip": "Shift hue angles, scale saturation, and adjust intensity cleanly.",
        "domain": "colorimetry",
        "func": col.hsi_vector_grading,
        "input_contract": "RGB",
        "output_contract": "RGB",
        "formula": r"H' = (H + \Delta H) \bmod 360^\circ, \quad S' = \text{clip}(S \cdot s_{scale}, 0, 1)",
        "summary": "Converts RGB to Hue-Saturation-Intensity, modifies chromatic vectors, and transforms back.",
        "example": "Shifts sky colors towards teal without distorting face brightness.",
        "viva_theory": "Converts RGB to Hue-Saturation-Intensity, modifies chromatic vectors, and transforms back.",
        "params": {
            "hue_shift": {"type": "float", "label": "Hue Angle Rotation (\u0394H \u00b0)", "default": 0.0, "min": -180.0, "max": 180.0, "step": 5.0},
            "sat_scale": {"type": "float", "label": "Saturation Scaling Factor", "default": 1.2, "min": 0.0, "max": 3.0, "step": 0.1},
            "intensity_offset": {"type": "float", "label": "Intensity (Brightness) Offset", "default": 0.0, "min": -100.0, "max": 100.0, "step": 5.0}
        }
    },
    "col_lift_gamma_gain": {
        "name": "Three-Way Color Balancer (Lift/Gamma/Gain)",
        "friendly_name": "Color Wheels (Shadows, Midtones, Highlights)",
        "simple_description": "Studio grading wheels to tint shadows, midtones, and highlights independently.",
        "tooltip": "Independent three-way color control for shadows (Lift), midtones (Gamma), and highlights (Gain).",
        "domain": "colorimetry",
        "func": col.three_way_lift_gamma_gain,
        "input_contract": "RGB",
        "output_contract": "RGB",
        "formula": r"I_{graded} = \text{Lift} \cdot (1 - I) + I^{1/\text{Gamma}} \cdot \text{Gain}",
        "summary": "DaVinci Resolve industry standard three-way tonal wheel calibration.",
        "example": "Tints shadow tones cold blue while preserving warm golden skin highlight reflections.",
        "viva_theory": "DaVinci Resolve industry standard three-way tonal wheel calibration.",
        "params": {
            "lift": {"type": "float", "label": "Shadows (Lift) Offset", "default": 0.05, "min": -0.5, "max": 0.5, "step": 0.02},
            "gamma": {"type": "float", "label": "Midtones (Gamma) Balance", "default": 1.10, "min": 0.5, "max": 2.0, "step": 0.05},
            "gain": {"type": "float", "label": "Highlights (Gain) Multiplier", "default": 1.10, "min": 0.5, "max": 2.0, "step": 0.05}
        }
    },
    "col_white_balance": {
        "name": "Von Kries Chromatic Adaptation",
        "friendly_name": "Fix White Balance (Warm / Cool Temp)",
        "simple_description": "Fixes unnatural yellow indoor bulbs or overly blue lighting so whites look truly white.",
        "tooltip": "Adjust color temperature (warm/cool) and tint (green/magenta).",
        "domain": "colorimetry",
        "func": col.von_kries_white_balance,
        "input_contract": "RGB",
        "output_contract": "RGB",
        "formula": r"\mathbf{C}' = \text{diag}(1/R_w, 1/G_w, 1/B_w) \cdot \mathbf{C}",
        "summary": "Removes unnatural environmental color casts using the Gray World assumption.",
        "example": "Corrects yellow tungsten incandescent or blueish fluorescent lighting in indoor photography.",
        "viva_theory": "Removes unnatural environmental color casts using the Gray World assumption.",
        "params": {
            "temperature": {"type": "float", "label": "Color Temperature", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05},
            "tint": {"type": "float", "label": "Green/Magenta Tint", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}
        }
    },
    "col_cmyk": {
        "name": "CMYK Subtractive Print Decomposition",
        "friendly_name": "Inspect Print Inks (CMYK Channels)",
        "simple_description": "Separates colors into Cyan, Magenta, Yellow, and Black ink plates used in professional book and magazine printing.",
        "tooltip": "Inspect commercial subtractive printing plates (Cyan, Magenta, Yellow, Black).",
        "domain": "colorimetry",
        "func": col.rgb_to_cmyk_visualization,
        "input_contract": "RGB",
        "output_contract": "GRAYSCALE",
        "formula": r"K = 1 - \max(R, G, B), \quad C = \frac{1 - R - K}{1 - K}",
        "summary": "Decomposes color into Cyan, Magenta, Yellow, and Key (Black) subtractive ink plates.",
        "example": "Essential prepress separation for commercial offset publishing and physical printing.",
        "viva_theory": "Decomposes color into Cyan, Magenta, Yellow, and Key (Black) subtractive ink plates.",
        "params": {
            "channel_view": {"type": "select", "label": "Active Subtractive Ink Plate", "default": "K", "options": ["C", "M", "Y", "K", "all"]}
        }
    },
    "col_rgb_matrix": {
        "name": "Multi-Channel Linear RGB Mixer",
        "friendly_name": "Control RGB Colors (Channel Mixer)",
        "simple_description": "Mix and boost individual Red, Green, and Blue channels for custom color correction or creative tints.",
        "tooltip": "Independently scale and offset Red, Green, and Blue color channels.",
        "domain": "colorimetry",
        "func": col.multi_channel_rgb_matrix,
        "input_contract": "RGB",
        "output_contract": "RGB",
        "formula": r"\mathbf{C}' = \mathbf{M}_{3\times3} \cdot \mathbf{C}",
        "summary": "Applies a customizable 3x3 channel mixing matrix to cross-blend spectral bands.",
        "example": "Simulates infrared false-color film looks and technical camera matrix profiles.",
        "viva_theory": "Applies a customizable 3x3 channel mixing matrix to cross-blend spectral bands.",
        "params": {
            "r_gain": {"type": "float", "label": "Red Channel Gain", "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1},
            "g_gain": {"type": "float", "label": "Green Channel Gain", "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1},
            "b_gain": {"type": "float", "label": "Blue Channel Gain", "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1},
            "r_offset": {"type": "int", "label": "Red Offset", "default": 0, "min": -100, "max": 100, "step": 5},
            "g_offset": {"type": "int", "label": "Green Offset", "default": 0, "min": -100, "max": 100, "step": 5},
            "b_offset": {"type": "int", "label": "Blue Offset", "default": 0, "min": -100, "max": 100, "step": 5}
        }
    },
    "col_vibrance": {
        "name": "Photoshop Vibrance & Smart Saturation",
        "friendly_name": "Vibrance & Smart Saturation",
        "simple_description": "Makes dull colors pop with life while intelligently protecting skin tones from turning orange.",
        "tooltip": "Boost muted colors safely without oversaturating skin tones.",
        "domain": "colorimetry",
        "func": col.vibrance_saturation,
        "input_contract": "RGB",
        "output_contract": "SAME",
        "formula": r"S' = S + (1 - S) \cdot \text{Vibrance}",
        "summary": "Selectively increases saturation of muted colors while preventing oversaturation of skin tones.",
        "example": "Makes landscapes and sunset skies pop naturally without making people look sunburned.",
        "viva_theory": "Selectively increases saturation of muted colors while preventing oversaturation of skin tones.",
        "params": {
            "vibrance": {"type": "float", "label": "Smart Vibrance Boost", "default": 0.50, "min": -1.0, "max": 1.5, "step": 0.05},
            "saturation": {"type": "float", "label": "Linear Saturation Scale", "default": 0.10, "min": -1.0, "max": 1.0, "step": 0.05}
        }
    },
    "col_sepia": {
        "name": "Antique Sepia Tone Tint",
        "friendly_name": "Vintage Warm Sepia Tone",
        "simple_description": "Gives modern digital pictures an authentic antique reddish-brown historical photo finish.",
        "tooltip": "Convert to warm antique sepia silver photographic tone.",
        "domain": "colorimetry",
        "func": col.sepia_vintage,
        "input_contract": "RGB",
        "output_contract": "SAME",
        "formula": r"\mathbf{RGB}' = \mathbf{M}_{sepia} \cdot \mathbf{RGB}",
        "summary": "Transforms full-color imagery into warm reddish-brown vintage silver photographic prints.",
        "example": "Recreates nostalgic, historical archive looks on modern digital photos.",
        "viva_theory": "Transforms full-color imagery into warm reddish-brown vintage silver photographic prints.",
        "params": {
            "strength": {"type": "float", "label": "Sepia Blend Strength", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}
        }
    },
    "col_split_toning": {
        "name": "DaVinci Resolve Split Toning",
        "friendly_name": "Split Toning (Tint Shadows & Highlights)",
        "simple_description": "Tints dark shadows one color (like cool teal) and bright highlights another (like warm gold).",
        "tooltip": "Apply complementary color tints to shadows and highlights separately.",
        "domain": "colorimetry",
        "func": col.split_toning,
        "input_contract": "RGB",
        "output_contract": "SAME",
        "formula": r"I' = I + (1 - L) \cdot \text{ShadowTint} + L \cdot \text{HighlightTint}",
        "summary": "Applies complementary color grading by tinting dark shadows and bright highlights independently.",
        "example": "Industry standard Hollywood look tinting shadows cool blue/teal and highlights warm amber/gold.",
        "viva_theory": "Applies complementary color grading by tinting dark shadows and bright highlights independently.",
        "params": {
            "shadow_hue": {"type": "float", "label": "Shadows Hue (°)", "default": 210.0, "min": 0.0, "max": 360.0, "step": 5.0},
            "shadow_sat": {"type": "float", "label": "Shadows Tint Saturation", "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05},
            "highlight_hue": {"type": "float", "label": "Highlights Hue (°)", "default": 40.0, "min": 0.0, "max": 360.0, "step": 5.0},
            "highlight_sat": {"type": "float", "label": "Highlights Tint Saturation", "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05}
        }
    },

    # Domain 5: Segmentation & structural
    "seg_sobel_balancer": {
        "name": "Sobel Directional Gradient Operator",
        "friendly_name": "Detect Edges (Sobel Outlines)",
        "simple_description": "Traces sharp boundaries and borders in the image using horizontal and vertical gradient detection.",
        "tooltip": "Calculate edge gradients to trace object boundaries.",
        "domain": "segmentation",
        "func": seg.directional_gradient_sobel,
        "input_contract": "ANY",
        "output_contract": "GRAYSCALE",
        "formula": r"M(x, y) = \sqrt{G_x^2 + G_y^2}, \quad \alpha(x, y) = \arctan(G_y / G_x)",
        "summary": "Computes first-order spatial gradients using 3x3 horizontal and vertical differential kernels.",
        "example": "Locates sharp contrast transitions indicating physical edges and outlines of objects.",
        "viva_theory": "Computes first-order spatial gradients using 3x3 horizontal and vertical differential kernels.",
        "params": {
            "alpha": {"type": "float", "label": "Horizontal Gx Weight", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
            "beta": {"type": "float", "label": "Vertical Gy Weight", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
            "ksize": {"type": "int", "label": "Sobel Aperture Size", "default": 3, "min": 3, "max": 7, "step": 2}
        }
    },
    "seg_prewitt": {
        "name": "Prewitt Compass Gradient Filter",
        "friendly_name": "Detect Edges (Prewitt Outlines)",
        "simple_description": "Highlights structural borders and transitions with balanced direction sensitivity.",
        "tooltip": "Detect horizontal and vertical outlines using uniform Prewitt kernels.",
        "domain": "segmentation",
        "func": seg.prewitt_gradient,
        "input_contract": "ANY",
        "output_contract": "GRAYSCALE",
        "formula": r"G_x = \begin{bmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \\ -1 & 0 & 1 \end{bmatrix} * f",
        "summary": "Computes edge magnitude with uniform weights (simpler than Sobel with less central emphasis).",
        "example": "Detects structural horizontal and vertical edges in low-complexity image processing pipelines.",
        "viva_theory": "Computes edge magnitude with uniform weights (simpler than Sobel with less central emphasis).",
        "params": {}
    },
    "seg_laplacian": {
        "name": "Laplacian Second-Derivative Filter",
        "friendly_name": "Fine Edge Outlines (Laplacian)",
        "simple_description": "Finds fine hairline cracks, delicate outlines, and pinpoint details regardless of their direction.",
        "tooltip": "Second-derivative filter to extract fine omnidirectional edge outlines.",
        "domain": "segmentation",
        "func": seg.laplacian_second_derivative,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}",
        "summary": "Measures rate of change of gradient (zero-crossings) to detect fine edges and rapid transitions.",
        "example": "Highlights fine outlines, hairline cracks, and isolated points regardless of direction.",
        "viva_theory": "Measures rate of change of gradient (zero-crossings) to detect fine edges and rapid transitions.",
        "params": {
            "ksize": {"type": "int", "label": "Laplacian Aperture", "default": 3, "min": 1, "max": 7, "step": 2}
        }
    },
    "seg_high_boost": {
        "name": "Unsharp Masking & High-Boost Filtering",
        "friendly_name": "High-Boost Sharpening",
        "simple_description": "Amplifies fine details and edges to make blurry scans and text punchy and readable.",
        "tooltip": "Subtract blurred image and add back amplified high-frequency details.",
        "domain": "segmentation",
        "func": seg.parametric_high_boost,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"f_{\text{hb}} = A \cdot f(x, y) - f_{\text{smooth}}(x, y)",
        "summary": "Subtracts blurred version from original to form edge mask, then adds it back with boost factor A.",
        "example": "Sharpens soft or slightly out-of-focus photographs and clarifies blurry text.",
        "viva_theory": "Subtracts blurred version from original to form edge mask, then adds it back with boost factor A.",
        "params": {
            "boost_a": {"type": "float", "label": "Boost Amplification (A)", "default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1},
            "ksize": {"type": "int", "label": "Blur Kernel Size", "default": 5, "min": 3, "max": 15, "step": 2}
        }
    },
    "seg_canny": {
        "name": "Canny Multi-Stage Edge Detector",
        "friendly_name": "Clean Line Drawing (Canny Edges)",
        "simple_description": "Produces clean, thin, connected line outlines of all objects, ideal for sketches and robot vision.",
        "tooltip": "Generate clean, single-pixel connected boundary contours.",
        "domain": "segmentation",
        "func": seg.canny_edge_detector,
        "input_contract": "ANY",
        "output_contract": "BINARY",
        "formula": r"\text{Canny}(f) = \text{Hysteresis}(\text{NonMaxSuppression}(\nabla G_\sigma * f))",
        "summary": "State-of-the-art optimal edge detector using Gaussian smoothing, non-maximum suppression, and dual-threshold hysteresis.",
        "example": "Extracts clean, single-pixel thin continuous boundaries for autonomous driving and object recognition.",
        "viva_theory": "State-of-the-art optimal edge detector using Gaussian smoothing, non-maximum suppression, and dual-threshold hysteresis.",
        "params": {
            "threshold_low": {"type": "int", "label": "Lower Hysteresis Cutoff", "default": 50, "min": 10, "max": 150, "step": 5},
            "threshold_high": {"type": "int", "label": "Upper Hysteresis Cutoff", "default": 150, "min": 60, "max": 255, "step": 5},
            "aperture_size": {"type": "int", "label": "Sobel Aperture Size", "default": 3, "min": 3, "max": 7, "step": 2}
        }
    },
    "seg_manual_thresh": {
        "name": "Global Fixed-Level Binarization",
        "friendly_name": "Black & White (Manual Cutoff)",
        "simple_description": "Converts image into pure crisp black and pure white based on a cutoff slider you control.",
        "tooltip": "Convert image into binary black and white using a manual threshold.",
        "domain": "segmentation",
        "func": seg.global_manual_threshold,
        "input_contract": "ANY",
        "output_contract": "BINARY",
        "formula": r"g(x, y) = \begin{cases} 255 & \text{if } f(x, y) \ge T \\ 0 & \text{if } f(x, y) < T \end{cases}",
        "summary": "Converts an image into strictly black and white by comparing each pixel against a fixed threshold T.",
        "example": "Separates high-contrast printed text from clear white paper.",
        "viva_theory": "Converts an image into strictly black and white by comparing each pixel against a fixed threshold T.",
        "params": {
            "threshold": {"type": "int", "label": "Cutoff Threshold (T)", "default": 128, "min": 0, "max": 255, "step": 1}
        }
    },
    "seg_otsu": {
        "name": "Otsu Automatic Variance Thresholding",
        "friendly_name": "Smart Black & White (Automatic Otsu)",
        "simple_description": "Automatically figures out the ideal cutoff point to turn document scans or photos into clean black and white.",
        "tooltip": "Statistically compute optimal black and white separation automatically.",
        "domain": "segmentation",
        "func": seg.otsu_automatic_threshold,
        "input_contract": "ANY",
        "output_contract": "BINARY",
        "formula": r"\sigma_B^2(k^*) = \max \frac{[\mu_T \omega(k) - \mu(k)]^2}{\omega(k)(1 - \omega(k))}",
        "summary": "Statistically calculates the optimal threshold that maximizes the variance between foreground and background classes.",
        "example": "Automatically binarizes document scans and microchip wafer inspections without manual tuning.",
        "viva_theory": "Statistically calculates the optimal threshold that maximizes the variance between foreground and background classes.",
        "params": {}
    },
    "seg_iterative_thresh": {
        "name": "Iterative Optimal Mean Thresholding",
        "friendly_name": "Smart Silhouette B&W (Iterative Mean)",
        "simple_description": "Finds the balanced cutoff point step-by-step for challenging backlit objects and silhouettes.",
        "tooltip": "Iteratively converge on the optimal foreground/background cutoff.",
        "domain": "segmentation",
        "func": seg.iterative_optimal_threshold,
        "input_contract": "ANY",
        "output_contract": "BINARY",
        "formula": r"T_{next} = \frac{1}{2}(\mu_1 + \mu_2)",
        "summary": "Iteratively divides pixels into foreground/background sets until the threshold converges.",
        "example": "Robust segmentation of backlit objects and silhouettes where histogram valleys are shallow.",
        "viva_theory": "Iteratively divides pixels into foreground/background sets until the threshold converges.",
        "params": {
            "delta_t": {"type": "float", "label": "Convergence Tolerance (\u0394T)", "default": 0.5, "min": 0.1, "max": 5.0, "step": 0.1}
        }
    },
    "seg_hough_lines": {
        "name": "Hough Line Transform",
        "friendly_name": "Find Straight Lines (Hough Lines)",
        "simple_description": "Automatically detects and draws straight lines, road lane markings, and architectural beams.",
        "tooltip": "Detect structural straight lines using mathematical Hough transform voting.",
        "domain": "segmentation",
        "func": seg.hough_line_transform,
        "input_contract": "ANY",
        "output_contract": "RGB",
        "formula": r"\rho = x \cos \theta + y \sin \theta",
        "summary": "Maps boundary points into (\u03c1, \u03b8) parameter space to detect straight lines via accumulator votes.",
        "example": "Detects roadway lane markings in driver assistance systems and structural building edges.",
        "viva_theory": "Maps boundary points into (\u03c1, \u03b8) parameter space to detect straight lines via accumulator votes.",
        "params": {
            "threshold_votes": {"type": "int", "label": "Accumulator Vote Threshold", "default": 80, "min": 20, "max": 200, "step": 5},
            "min_line_len": {"type": "float", "label": "Minimum Line Length", "default": 50.0, "min": 10.0, "max": 200.0, "step": 5.0},
            "max_line_gap": {"type": "float", "label": "Maximum Co-linear Gap", "default": 10.0, "min": 1.0, "max": 50.0, "step": 1.0},
            "draw_on_source": {"type": "bool", "label": "Overlay Lines on Source Image", "default": True}
        }
    },
    "seg_adaptive_thresh": {
        "name": "Local Adaptive Gaussian Binarization",
        "friendly_name": "Document Scanner B&W (Shadow-Proof)",
        "simple_description": "Turns photos of paper into clean black-and-white text even when shadows or uneven lamps fall across the page.",
        "tooltip": "Binarize text cleanly under uneven lighting or casting shadows.",
        "domain": "segmentation",
        "func": seg.adaptive_threshold,
        "input_contract": "ANY",
        "output_contract": "BINARY",
        "formula": r"T(x, y) = \text{Gaussian}(S_{xy}) - C",
        "summary": "Calculates local threshold per pixel window to binarize images with non-uniform illumination.",
        "example": "Extracts clean text from document photos taken under casting shadows or uneven table lamps.",
        "viva_theory": "Calculates local threshold per pixel window to binarize images with non-uniform illumination.",
        "params": {
            "block_size": {"type": "int", "label": "Local Neighborhood Block", "default": 11, "min": 3, "max": 45, "step": 2},
            "c": {"type": "int", "label": "Constant Subtraction (C)", "default": 2, "min": -10, "max": 20, "step": 1},
            "method": {"type": "select", "label": "Weighting Function", "default": "gaussian", "options": ["gaussian", "mean"]}
        }
    },
    "seg_hough_circles": {
        "name": "Hough Circle Transform",
        "friendly_name": "Find Round Circles (Hough Circles)",
        "simple_description": "Detects circular objects like coins, pills, or eyes and marks their exact outlines.",
        "tooltip": "Identify and outline circular objects and radii in the image.",
        "domain": "segmentation",
        "func": seg.hough_circle_transform,
        "input_contract": "ANY",
        "output_contract": "RGB",
        "formula": r"(x - a)^2 + (y - b)^2 = r^2",
        "summary": "Detects circular objects, coins, and pupils via gradient direction Hough voting.",
        "example": "Automatically counts circular pills in blister packs or detects iris boundaries in biometrics.",
        "viva_theory": "Detects circular objects, coins, and pupils via gradient direction Hough voting.",
        "params": {
            "min_dist": {"type": "int", "label": "Min Distance Between Centers", "default": 35, "min": 10, "max": 150, "step": 5},
            "param1": {"type": "int", "label": "Canny Upper Hysteresis", "default": 100, "min": 30, "max": 200, "step": 5},
            "param2": {"type": "int", "label": "Accumulator Vote Threshold", "default": 30, "min": 15, "max": 80, "step": 2},
            "min_radius": {"type": "int", "label": "Min Circle Radius", "default": 10, "min": 5, "max": 80, "step": 5},
            "max_radius": {"type": "int", "label": "Max Circle Radius", "default": 120, "min": 20, "max": 250, "step": 5}
        }
    },
    "seg_harris_corners": {
        "name": "Harris & Stephens Corner Detection",
        "friendly_name": "Find Corner Points (Harris Corners)",
        "simple_description": "Pinpoints sharp corners and intersection points, used in panorama stitching and motion tracking.",
        "tooltip": "Detect rotation-invariant corner keypoints across the image.",
        "domain": "segmentation",
        "func": seg.harris_corners,
        "input_contract": "ANY",
        "output_contract": "RGB",
        "formula": r"R = \det(\mathbf{M}) - k \cdot (\text{trace}(\mathbf{M}))^2",
        "summary": "Detects rotation-invariant corner keypoints where gradient changes along all directions.",
        "example": "Used for panoramic image stitching, motion tracking, and camera visual odometry.",
        "viva_theory": "Detects rotation-invariant corner keypoints where gradient changes along all directions.",
        "params": {
            "block_size": {"type": "int", "label": "Corner Neighborhood Size", "default": 2, "min": 2, "max": 6, "step": 1},
            "ksize": {"type": "int", "label": "Sobel Aperture Size", "default": 3, "min": 3, "max": 7, "step": 2},
            "k": {"type": "float", "label": "Harris Sensitivity Constant", "default": 0.04, "min": 0.01, "max": 0.08, "step": 0.01},
            "threshold": {"type": "float", "label": "Response Peak Threshold", "default": 0.01, "min": 0.001, "max": 0.05, "step": 0.002}
        }
    },

    # Domain 6: Morphology
    "morph_dilate": {
        "name": "Morphological Dilation",
        "friendly_name": "Thicken White Regions (Dilate)",
        "simple_description": "Expands white shapes, makes thin letters bolder, and bridges small broken gaps.",
        "tooltip": "Grow white foreground boundaries and bridge broken letters.",
        "domain": "morphology",
        "func": morph.morphological_dilate,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"A \oplus B = \{ z \mid (\hat{B})_z \cap A \ne \emptyset \}",
        "summary": "Expands foreground object boundaries and bridges small interior holes.",
        "example": "Connects broken characters in degraded OCR text scans.",
        "viva_theory": "Expands foreground object boundaries and bridges small interior holes.",
        "params": {
            "ksize": {"type": "int", "label": "Structuring Element Size", "default": 3, "min": 3, "max": 15, "step": 2},
            "shape": {"type": "select", "label": "SE Geometric Shape", "default": "rect", "options": ["rect", "ellipse", "cross"]}
        }
    },
    "morph_erode": {
        "name": "Morphological Erosion",
        "friendly_name": "Thin & Shrink Shapes (Erode)",
        "simple_description": "Trims away outer edges of white shapes, making thick lines thinner and wiping away small dust flecks.",
        "tooltip": "Shrink white foreground boundaries and remove fine spurs.",
        "domain": "morphology",
        "func": morph.morphological_erode,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"A \ominus B = \{ z \mid (B)_z \subseteq A \}",
        "summary": "Shrinks object boundaries and strips away thin foreground protrusions and noise spurs.",
        "example": "Eliminates tiny dust specks and separates touching foreground items.",
        "viva_theory": "Shrinks object boundaries and strips away thin foreground protrusions and noise spurs.",
        "params": {
            "ksize": {"type": "int", "label": "Structuring Element Size", "default": 3, "min": 3, "max": 15, "step": 2},
            "shape": {"type": "select", "label": "SE Geometric Shape", "default": "rect", "options": ["rect", "ellipse", "cross"]}
        }
    },
    "morph_open": {
        "name": "Morphological Opening (Filter Spikes)",
        "friendly_name": "Clean Specks & Small Spots (Opening)",
        "simple_description": "Wipes away tiny bright noise specks and separates objects that are lightly touching.",
        "tooltip": "Erode then dilate to eliminate tiny isolated bright noise spots.",
        "domain": "morphology",
        "func": morph.morphological_open,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"A \circ B = (A \ominus B) \oplus B",
        "summary": "Erosion followed by dilation; eliminates isolated bright noise spikes while preserving shape.",
        "example": "Cleans salt noise and separates connected biological cells in microscopy.",
        "viva_theory": "Erosion followed by dilation; eliminates isolated bright noise spikes while preserving shape.",
        "params": {
            "ksize": {"type": "int", "label": "Structuring Element Size", "default": 3, "min": 3, "max": 15, "step": 2},
            "shape": {"type": "select", "label": "SE Geometric Shape", "default": "rect", "options": ["rect", "ellipse", "cross"]}
        }
    },
    "morph_close": {
        "name": "Morphological Closing (Fill Gaps)",
        "friendly_name": "Fill Small Holes & Gaps (Closing)",
        "simple_description": "Fills in tiny black holes inside objects and bridges cracks in broken letters.",
        "tooltip": "Dilate then erode to seal narrow gaps and fill interior dark holes.",
        "domain": "morphology",
        "func": morph.morphological_close,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"A \bullet B = (A \oplus B) \ominus B",
        "summary": "Dilation followed by erosion; fills narrow gaps, bridges breaks, and fills dark holes.",
        "example": "Heals broken boundary contours and seals gaps in road segmentations.",
        "viva_theory": "Dilation followed by erosion; fills narrow gaps, bridges breaks, and fills dark holes.",
        "params": {
            "ksize": {"type": "int", "label": "Structuring Element Size", "default": 3, "min": 3, "max": 15, "step": 2},
            "shape": {"type": "select", "label": "SE Geometric Shape", "default": "rect", "options": ["rect", "ellipse", "cross"]}
        }
    },
    "morph_gradient": {
        "name": "Morphological Gradient",
        "friendly_name": "Hollow Outline (Morphological Edge)",
        "simple_description": "Creates a hollow outline of shapes by subtracting the shrunk core from the expanded edge.",
        "tooltip": "Generate clean perimeter outlines using structural expansion and contraction.",
        "domain": "morphology",
        "func": morph.morphological_gradient,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"G(A) = (A \oplus B) - (A \ominus B)",
        "summary": "Subtracts the eroded image from the dilated image to isolate the boundary outline.",
        "example": "Reveals structural object outlines without needing derivative Sobel filters.",
        "viva_theory": "Subtracts the eroded image from the dilated image to isolate the boundary outline.",
        "params": {
            "ksize": {"type": "int", "label": "Structuring Element Size", "default": 3, "min": 3, "max": 15, "step": 2},
            "shape": {"type": "select", "label": "SE Geometric Shape", "default": "rect", "options": ["rect", "ellipse", "cross"]}
        }
    },
    "morph_tophat": {
        "name": "White Top-Hat Transform",
        "friendly_name": "Extract Bright Highlights (Top-Hat)",
        "simple_description": "Isolates elements that are brighter than their immediate background, like stars in the night sky.",
        "tooltip": "Extract bright details and features on unevenly lit backgrounds.",
        "domain": "morphology",
        "func": morph.morphological_tophat,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"T_{\text{hat}}(A) = A - (A \circ B)",
        "summary": "Subtracts opened image from original; extracts elements brighter than their surroundings.",
        "example": "Extracts stars against nighttime sky and corrects non-uniform microscope lighting.",
        "viva_theory": "Subtracts opened image from original; extracts elements brighter than their surroundings.",
        "params": {
            "ksize": {"type": "int", "label": "Structuring Element Size", "default": 9, "min": 3, "max": 31, "step": 2},
            "shape": {"type": "select", "label": "SE Geometric Shape", "default": "rect", "options": ["rect", "ellipse", "cross"]}
        }
    },
    "morph_blackhat": {
        "name": "Black-Hat (Bottom-Hat) Transform",
        "friendly_name": "Extract Dark Grooves (Black-Hat)",
        "simple_description": "Reveals dark sunken features like microscopic cracks, dark text, or blood vessels.",
        "tooltip": "Extract dark features and depressions against brighter backgrounds.",
        "domain": "morphology",
        "func": morph.morphological_blackhat,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"B_{\text{hat}}(A) = (A \bullet B) - A",
        "summary": "Subtracts original from closed image; isolates dark structures and depressions.",
        "example": "Detects microscopic cracks on concrete surfaces and fine retinal blood vessels.",
        "viva_theory": "Subtracts original from closed image; isolates dark structures and depressions.",
        "params": {
            "ksize": {"type": "int", "label": "Structuring Element Size", "default": 9, "min": 3, "max": 31, "step": 2},
            "shape": {"type": "select", "label": "SE Geometric Shape", "default": "rect", "options": ["rect", "ellipse", "cross"]}
        }
    },
    "morph_boundary": {
        "name": "Boundary Extraction Operator",
        "friendly_name": "Perimeter Wireframe (Boundary)",
        "simple_description": "Strips away the inside of solid shapes, leaving only a clean one-pixel perimeter line.",
        "tooltip": "Extract the outer perimeter wireframe contour of shapes.",
        "domain": "morphology",
        "func": morph.boundary_extraction,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"\beta(A) = A - (A \ominus B)",
        "summary": "Subtracts internal eroded core from foreground set to produce clean perimeter lines.",
        "example": "Extracts contour wireframes of segmented objects.",
        "viva_theory": "Subtracts internal eroded core from foreground set to produce clean perimeter lines.",
        "params": {
            "ksize": {"type": "int", "label": "Structuring Element Size", "default": 3, "min": 3, "max": 9, "step": 2},
            "shape": {"type": "select", "label": "SE Geometric Shape", "default": "rect", "options": ["rect", "ellipse", "cross"]}
        }
    },
    "morph_hit_and_miss": {
        "name": "Hit-or-Miss Transform",
        "friendly_name": "Spot Corner & End Points (Hit-or-Miss)",
        "simple_description": "Finds specific pixel geometries like line tips, corners, or isolated single dots.",
        "tooltip": "Locate specific foreground/background pixel geometries.",
        "domain": "morphology",
        "func": morph.hit_and_miss_transform,
        "input_contract": "BINARY",
        "output_contract": "BINARY",
        "formula": r"A \circledast B = (A \ominus B_1) \cap (A^c \ominus B_2)",
        "summary": "Applies dual structuring elements to detect specific foreground and background pixel patterns.",
        "example": "Finds corners, isolated points, and line endpoints in binary images.",
        "viva_theory": "Applies dual structuring elements to detect specific foreground and background pixel patterns.",
        "params": {
            "target": {"type": "select", "label": "Pattern Target", "default": "isolated_points", "options": ["isolated_points", "corners"]}
        }
    },

    # Domain 7: Creative FX & Photoshop studio
    "art_pencil_sketch": {
        "name": "Hand-Drawn Pencil Sketch",
        "friendly_name": "Pencil Sketch / Hand Drawing",
        "simple_description": "Turns photos into realistic graphite pencil drawings or colored sketch illustrations.",
        "tooltip": "Convert image into realistic graphite or colored pencil artwork.",
        "domain": "artistic",
        "func": art.pencil_sketch,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"\text{Sketch} = \min(255, (\text{Gray} \cdot 256) / (255 - \text{Blur} + 1))",
        "summary": "Blends inverted Gaussian blur with grayscale raster via Color Dodge to create realistic graphite sketches.",
        "example": "Transforms portrait photos and architectural views into fine-art hand-drawn illustrations.",
        "viva_theory": "Blends inverted Gaussian blur with grayscale raster via Color Dodge to create realistic graphite sketches.",
        "params": {
            "ksize": {"type": "int", "label": "Pencil Stroke Texture Blur", "default": 21, "min": 5, "max": 51, "step": 2},
            "color_mode": {"type": "select", "label": "Sketch Medium", "default": "pencil_gray", "options": ["pencil_gray", "pencil_color"]}
        }
    },
    "art_cartoonify": {
        "name": "Cel-Shaded Cartoonify Effect",
        "friendly_name": "Comic Book / Cartoon Art",
        "simple_description": "Flattens colors and draws bold black ink outlines to turn photos into graphic comic book art.",
        "tooltip": "Flatten colors with bilateral filter and overlay ink outlines for cartoon effect.",
        "domain": "artistic",
        "func": art.cartoonify_filter,
        "input_contract": "RGB",
        "output_contract": "SAME",
        "formula": r"\text{Cartoon} = \text{Bilateral}(I) \land \text{AdaptiveEdges}(I)",
        "summary": "Flattens photo colors via multi-pass bilateral filtering and overlays crisp black ink line art.",
        "example": "Gives everyday photographs an animated graphic novel or cartoon aesthetic.",
        "viva_theory": "Flattens photo colors via multi-pass bilateral filtering and overlays crisp black ink line art.",
        "params": {
            "num_bilateral": {"type": "int", "label": "Color Flattening Passes", "default": 3, "min": 1, "max": 6, "step": 1},
            "d": {"type": "int", "label": "Bilateral Spatial Window", "default": 9, "min": 5, "max": 15, "step": 2},
            "sigma_color": {"type": "float", "label": "Color Smoothing Bandwidth", "default": 75.0, "min": 25.0, "max": 150.0, "step": 5.0},
            "edge_block_size": {"type": "int", "label": "Ink Line Thickness Block", "default": 9, "min": 3, "max": 15, "step": 2}
        }
    },
    "art_emboss": {
        "name": "3D Directional Bas-Relief Emboss",
        "friendly_name": "3D Stamped Metal Emboss",
        "simple_description": "Gives your image a raised 3D texture, making it look like a stamped metal coin or stone carving.",
        "tooltip": "Simulate directional 3D relief lighting on metallic surfaces.",
        "domain": "artistic",
        "func": art.emboss_filter,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"\text{Emboss} = (I * K_{directional}) + 128",
        "summary": "Convolves image with a directional lighting gradient mask to simulate metallic 3D relief.",
        "example": "Creates stamped metal badge looks or tactile stone engraving effects on logos.",
        "viva_theory": "Convolves image with a directional lighting gradient mask to simulate metallic 3D relief.",
        "params": {
            "azimuth": {"type": "float", "label": "Light Angle (°)", "default": 45.0, "min": 0.0, "max": 360.0, "step": 15.0},
            "depth": {"type": "float", "label": "Relief Depth Multiplier", "default": 1.2, "min": 0.2, "max": 3.0, "step": 0.1}
        }
    },
    "art_vignette": {
        "name": "Radial Optical Lens Vignette",
        "friendly_name": "Dark / Light Border Vignette",
        "simple_description": "Softly darkens or lightens the outer borders to draw all viewer attention directly to the center subject.",
        "tooltip": "Apply smooth radial edge falloff to draw focus to the subject.",
        "domain": "artistic",
        "func": art.vignette_filter,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"V(r) = 1 - \text{intensity} \cdot (r / R_{\max})^2",
        "summary": "Applies quadratic radial falloff shading to draw viewer attention to the central subject.",
        "example": "Simulates classic Leica 35mm street photography lens shading on portraits.",
        "viva_theory": "Applies quadratic radial falloff shading to draw viewer attention to the central subject.",
        "params": {
            "radius": {"type": "float", "label": "Spotlight Radius", "default": 0.75, "min": 0.2, "max": 1.5, "step": 0.05},
            "intensity": {"type": "float", "label": "Vignette Darkness", "default": 0.60, "min": 0.1, "max": 1.0, "step": 0.05}
        }
    },
    "art_tilt_shift": {
        "name": "Tilt-Shift Miniature Blur",
        "friendly_name": "Toy Miniature Effect (Tilt-Shift)",
        "simple_description": "Blurs the top and bottom of your photo so real-world cities and vehicles look like tiny toy miniatures.",
        "tooltip": "Simulate shallow depth-of-field to make cityscapes look like miniature dioramas.",
        "domain": "artistic",
        "func": art.tilt_shift_filter,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"I_{tilt} = I \cdot \alpha(y) + I_{blur} \cdot (1 - \alpha(y))",
        "summary": "Simulates miniature photography by maintaining a sharp focal band and blurring outer margins.",
        "example": "Makes real cityscapes, cars, and railway scenes look like tiny tabletop toy dioramas.",
        "viva_theory": "Simulates miniature photography by maintaining a sharp focal band and blurring outer margins.",
        "params": {
            "focus_y": {"type": "float", "label": "Focal Band Center (Y)", "default": 0.50, "min": 0.1, "max": 0.9, "step": 0.05},
            "focus_height": {"type": "float", "label": "Sharp Focal Band Width", "default": 0.25, "min": 0.05, "max": 0.50, "step": 0.05},
            "blur_amount": {"type": "int", "label": "Defocus Blur Strength", "default": 17, "min": 5, "max": 45, "step": 2}
        }
    },

    # Domain 8: Compression & redundancy
    "comp_dct_jpeg": {
        "name": "Academic DCT 8x8 Block JPEG Simulation",
        "friendly_name": "JPEG Compression Simulator",
        "simple_description": "Lets you slide quality down from 100% to 1% to see classic JPEG 8x8 block artifacts and file savings.",
        "tooltip": "Simulate standard 8x8 DCT JPEG compression blocks and quality degradation.",
        "domain": "compression",
        "func": comp.dct_block_jpeg_simulation,
        "input_contract": "ANY",
        "output_contract": "SAME",
        "formula": r"F(u, v) = \frac{1}{4} C(u)C(v) \sum_{x=0}^{7}\sum_{y=0}^{7} f(x,y)\cos\frac{(2x+1)u\pi}{16}\cos\frac{(2y+1)v\pi}{16}",
        "summary": "Textbook 8x8 block Discrete Cosine Transform (DCT) JPEG baseline compression with psychovisual quantization matrix scaling.",
        "example": "Demonstrates lossy 8x8 blocking artifacts and high-frequency coefficient truncation as quality is lowered from 100% down to 5%.",
        "viva_theory": "Forward 2D-DCT transforms spatial intensity correlations into orthogonal spatial frequencies. High frequencies are aggressively quantized based on the Human Visual System contrast sensitivity function.",
        "params": {
            "quality": {"type": "int", "label": "JPEG Quality Factor (1-100)", "default": 35, "min": 1, "max": 100, "step": 1},
            "block_size": {"type": "int", "label": "DCT Block Dimension (px)", "default": 8, "min": 8, "max": 16, "step": 8}
        }
    },
    "comp_quantization": {
        "name": "Paletted Color Quantization (PNG-8 / GIF)",
        "friendly_name": "Color Palette Reducer (GIF Style)",
        "simple_description": "Reduces 16 million colors down to 2-256 indexed colors with retro dithered dots, like vintage GIFs.",
        "tooltip": "Reduce color palette to 2-256 colors with optional Floyd-Steinberg dithering.",
        "domain": "compression",
        "func": comp.paletted_color_quantization,
        "input_contract": "ANY",
        "output_contract": "RGB",
        "formula": r"C_{quant} = \text{argmin}_{c \in \mathcal{P}} \|C_{in} - c\|^2",
        "summary": "Reduces 24-bit full-color images (16.7M colors) into 2 to 256 indexed palette colors via Median Cut clustering and Floyd-Steinberg error diffusion.",
        "example": "Simulates classic GIF / 8-bit PNG palette compression with retro dithered color gradients.",
        "viva_theory": "Paletted color quantization clusters RGB color space into optimal Voronoi centroids, replacing 24-bit RGB pixel representations with compact 8-bit palette indices.",
        "params": {
            "num_colors": {"type": "int", "label": "Color Palette Size", "default": 32, "min": 2, "max": 256, "step": 2},
            "dither": {"type": "bool", "label": "Floyd-Steinberg Dithering", "default": True}
        }
    }
}

