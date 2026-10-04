"""
Generates synthetic standard reference images for ImageLab Pro.
Creates 4 diverse test benchmarks:
1. 'Chroma Chart & Natural Scene' (for Colorimetry & Radiometric testing)
2. 'Siemens Star & Frequency Target' (for Geometry, Sampling, & Aliasing testing)
3. 'Degraded Text Document' (for High-Boost, Otsu Binarization, & Morphology testing)
4. 'Geometric Shapes & Edges' (for Hough, Sobel, Canny, & Corner testing)
"""

import os
import cv2
import numpy as np


def generate_sample_images(output_dir: str = "assets/sample_images"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Color Palette & Natural Scene (512x512)
    img_color = np.zeros((512, 512, 3), dtype=np.uint8)
    # Sky gradient (top half)
    for y in range(256):
        b = int(240 - y * 0.4)
        g = int(180 - y * 0.3)
        r = int(120 - y * 0.2)
        img_color[y, :] = [r, g, b]

    # Warm sun
    cv2.circle(img_color, (140, 110), 45, (255, 220, 80), -1)

    # Rolling hills (bottom half)
    for x in range(512):
        h1 = int(250 + 40 * np.sin(x / 45.0))
        img_color[h1:, x] = [34, 139, 34]
        h2 = int(320 + 30 * np.sin((x + 100) / 60.0))
        img_color[h2:, x] = [46, 110, 46]

    # Color checker patches along the bottom
    patch_w = 512 // 6
    colors = [
        [220, 20, 60],   # Crimson Red
        [30, 144, 255],  # Dodger Blue
        [50, 205, 50],   # Lime Green
        [255, 215, 0],   # Gold Yellow
        [186, 85, 211],  # Medium Orchid
        [245, 245, 245]  # White
    ]
    for i, col in enumerate(colors):
        cv2.rectangle(img_color, (i * patch_w, 440), ((i + 1) * patch_w, 512), col, -1)
        cv2.rectangle(img_color, (i * patch_w, 440), ((i + 1) * patch_w, 512), (30, 30, 30), 2)

    cv2.imwrite(os.path.join(output_dir, "chroma_studio.png"), cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR))

    # 2. Siemens Star & Frequency Resolution Target (512x512)
    img_star = np.ones((512, 512), dtype=np.uint8) * 255
    cx, cy = 256, 256
    num_spokes = 36
    for i in range(num_spokes):
        if i % 2 == 0:
            a1 = (i / num_spokes) * 2 * np.pi
            a2 = ((i + 1) / num_spokes) * 2 * np.pi
            pts = np.array([
                [cx, cy],
                [int(cx + 240 * np.cos(a1)), int(cy + 240 * np.sin(a1))],
                [int(cx + 240 * np.cos(a2)), int(cy + 240 * np.sin(a2))]
            ], dtype=np.int32)
            cv2.fillPoly(img_star, [pts], 0)

    # Concentric circles for frequency inspection
    for r in range(30, 240, 25):
        cv2.circle(img_star, (cx, cy), r, 128, 2)

    # Grayscale linear ramp on top
    for x in range(512):
        val = int((x / 512.0) * 255)
        img_star[0:30, x] = val

    cv2.imwrite(os.path.join(output_dir, "resolution_target.png"), img_star)

    # 3. Degraded Document Text (512x512)
    img_doc = np.ones((512, 512), dtype=np.uint8) * 235
    # Uneven background illumination (vignette / shadow across page)
    y_coords, x_coords = np.mgrid[0:512, 0:512]
    shading = (180 + 70 * (x_coords / 512.0) - 60 * (y_coords / 512.0)).astype(np.float32)
    img_doc = np.clip(img_doc.astype(np.float32) * (shading / 255.0), 0, 255).astype(np.uint8)

    # Render simulated text lines
    lines = [
        "IMAGELAB PRO - BENCHMARK TEST",
        "Digital Image Processing Laboratory",
        "Testing Otsu Binarization & Canny Edge Link",
        "Morphological Probes: Dilate, Erode, Open, Close",
        "Spatial Quantization and Sampling Criterion",
        "Directional Gradient Balancer Gx, Gy",
        "Non-Destructive Node Pipeline Execution",
        "Academic Unit 1 to Unit 6 Complete Validation"
    ]
    for i, line in enumerate(lines):
        cv2.putText(img_doc, line, (35, 60 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)

    # Add small impulse dust spots
    for _ in range(120):
        rx, ry = np.random.randint(20, 490), np.random.randint(20, 490)
        cv2.circle(img_doc, (rx, ry), np.random.randint(1, 3), 15, -1)

    cv2.imwrite(os.path.join(output_dir, "document_ocr.png"), img_doc)

    # 4. Geometric Shapes & Polygons (512x512 RGB)
    img_geom = np.zeros((512, 512, 3), dtype=np.uint8) + 40
    # Prominent lines for Hough transform
    cv2.line(img_geom, (40, 60), (470, 60), (0, 220, 255), 3)
    cv2.line(img_geom, (60, 40), (60, 470), (255, 100, 200), 3)
    cv2.line(img_geom, (80, 450), (460, 90), (100, 255, 100), 3)

    # Rectangles & circles
    cv2.rectangle(img_geom, (120, 160), (280, 320), (240, 200, 50), -1)
    cv2.circle(img_geom, (380, 320), 75, (255, 60, 60), -1)

    cv2.imwrite(os.path.join(output_dir, "geometry_shapes.png"), cv2.cvtColor(img_geom, cv2.COLOR_RGB2BGR))
    print(f"Sample test images generated successfully in {output_dir}")


if __name__ == "__main__":
    generate_sample_images()
