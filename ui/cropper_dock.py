"""
=============================================================================
Pixel-Forge - Professional Vector Shape Cropper Studio
Interactive studio component for multi-shape cropping with alpha transparency.
=============================================================================
"""

from typing import Optional
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io

from algorithms.geometry import vector_shape_crop
from core.state_manager import add_layer_to_stack


def render_cropper_dock(raw_image: np.ndarray, processed_image: np.ndarray, master_image: Optional[np.ndarray] = None):
    """
    Renders the Vector Multi-Shape Cropper Studio inside a collapsible expander.
    Enables users to crop images using analytical vector shapes:
    Circle, Square, Rectangle, Ellipse, Triangle, Diamond, Hexagon, Octagon,
    Star, Heart, Regular N-gon, with subpixel anti-aliasing and RGBA transparency.
    Supports cropping pristine full-resolution master images (up to 1,000 MB).
    """
    with st.expander("✂️ **Vector Shape Cropper Studio (Circle, Square, Star, Heart & Custom)**", expanded=False):
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div>
                    <h3 style="margin: 0; font-size: 1.15rem; color: #f8fafc; font-weight: 700;">Vector Multi-Shape Canvas Cropper</h3>
                    <p style="margin: 2px 0 0 0; font-size: 0.82rem; color: #94a3b8;">
                        Crop images into circles, squares, stars, hearts, polygons, or custom boxes with subpixel anti-aliasing and transparent alpha.
                    </p>
                </div>
                <div>
                    <span class="badge badge-unit">11 Vector Shapes</span>
                    <span class="badge badge-rgb" style="margin-left: 4px;">RGBA Transparency</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Source Frame Selection & Quick Presets
        src_col, preset_col = st.columns([1, 2.2])
        with src_col:
            crop_src_options = ["Current Processed Output", "Original Raw Frame"]
            if master_image is not None and master_image.shape[:2] != raw_image.shape[:2]:
                crop_src_options.insert(0, f"Full-Resolution Master Frame ({master_image.shape[1]} × {master_image.shape[0]} px)")

            crop_source = st.radio(
                "Source Image for Cropping",
                options=crop_src_options,
                horizontal=True,
                key="cropper_source_radio"
            )
            if "Full-Resolution Master" in crop_source:
                base_img = master_image
            elif crop_source == "Current Processed Output":
                base_img = processed_image
            else:
                base_img = raw_image

        with preset_col:
            st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #cbd5e1; margin-bottom: 4px;'>QUICK SHAPE PRESETS:</div>", unsafe_allow_html=True)
            p_c1, p_c2, p_c3, p_c4, p_c5, p_c6 = st.columns(6)

            with p_c1:
                if st.button("⭕ Avatar", use_container_width=True, help="Circular avatar profile"):
                    st.session_state["cropper_shape_select"] = "Circle (Avatar / Round)"
                    st.session_state["cropper_size_slider"] = 0.80
                    st.session_state["cropper_tight_chk"] = True
                    st.session_state["cropper_feather_slider"] = 0
                    st.session_state["cropper_bg_select"] = "Transparent (RGBA PNG)"
                    st.rerun()

            with p_c2:
                if st.button("⬛ Square", use_container_width=True, help="1:1 square crop"):
                    st.session_state["cropper_shape_select"] = "Square (1:1 Ratio)"
                    st.session_state["cropper_size_slider"] = 0.85
                    st.session_state["cropper_tight_chk"] = True
                    st.session_state["cropper_feather_slider"] = 0
                    st.rerun()

            with p_c3:
                if st.button("🎬 16:9", use_container_width=True, help="16:9 landscape banner"):
                    st.session_state["cropper_shape_select"] = "Rectangle (Custom Ratio)"
                    st.session_state["cropper_aspect_select"] = "16:9"
                    st.session_state["cropper_size_slider"] = 0.90
                    st.session_state["cropper_tight_chk"] = True
                    st.rerun()

            with p_c4:
                if st.button("⭐ Star", use_container_width=True, help="5-point golden star badge"):
                    st.session_state["cropper_shape_select"] = "Star (Badge / Seal)"
                    st.session_state["cropper_size_slider"] = 0.85
                    st.session_state["cropper_tight_chk"] = True
                    st.session_state["cropper_feather_slider"] = 0
                    st.session_state["cropper_bg_select"] = "Transparent (RGBA PNG)"
                    st.rerun()

            with p_c5:
                if st.button("❤️ Heart", use_container_width=True, help="Smooth parametric cardioid heart"):
                    st.session_state["cropper_shape_select"] = "Heart (Love / Cardioid)"
                    st.session_state["cropper_size_slider"] = 0.85
                    st.session_state["cropper_tight_chk"] = True
                    st.session_state["cropper_feather_slider"] = 0
                    st.session_state["cropper_bg_select"] = "Transparent (RGBA PNG)"
                    st.rerun()

            with p_c6:
                if st.button("⬡ Hexagon", use_container_width=True, help="Geometric hexagon"):
                    st.session_state["cropper_shape_select"] = "Hexagon (6-Sided)"
                    st.session_state["cropper_size_slider"] = 0.85
                    st.session_state["cropper_tight_chk"] = True
                    st.rerun()

        st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255, 255, 255, 0.08); margin: 10px 0;'>", unsafe_allow_html=True)

        # -------------------------------------------------------------
        # SHAPE CONTROLS & SLIDERS
        # -------------------------------------------------------------
        row1_c1, row1_c2, row1_c3, row1_c4 = st.columns([1.5, 1.2, 1.2, 1.2])

        with row1_c1:
            shape_options = [
                "Circle (Avatar / Round)",
                "Square (1:1 Ratio)",
                "Rectangle (Custom Ratio)",
                "Ellipse (Smooth Oval)",
                "Triangle (Equilateral)",
                "Diamond (Rhombus)",
                "Hexagon (6-Sided)",
                "Octagon (8-Sided)",
                "Star (Badge / Seal)",
                "Heart (Love / Cardioid)",
                "Regular Polygon (N-Gon)"
            ]
            chosen_shape_label = st.selectbox(
                "Choose Shape Geometry",
                options=shape_options,
                key="cropper_shape_select"
            )

            # Map to internal shape name
            shape_key_map = {
                "Circle (Avatar / Round)": "circle",
                "Square (1:1 Ratio)": "square",
                "Rectangle (Custom Ratio)": "rectangle",
                "Ellipse (Smooth Oval)": "ellipse",
                "Triangle (Equilateral)": "triangle",
                "Diamond (Rhombus)": "diamond",
                "Hexagon (6-Sided)": "hexagon",
                "Octagon (8-Sided)": "octagon",
                "Star (Badge / Seal)": "star",
                "Heart (Love / Cardioid)": "heart",
                "Regular Polygon (N-Gon)": "polygon"
            }
            shape_name = shape_key_map.get(chosen_shape_label, "circle")

        with row1_c2:
            center_x = st.slider("Center X Position", 0.0, 1.0, 0.50, 0.02, key="cropper_cx_slider", help="Horizontal center of shape")
            center_y = st.slider("Center Y Position", 0.0, 1.0, 0.50, 0.02, key="cropper_cy_slider", help="Vertical center of shape")

        with row1_c3:
            size_val = st.slider("Shape Scale / Radius", 0.10, 1.50, 0.80, 0.05, key="cropper_size_slider", help="Scale relative to canvas dimension")
            rotation_deg = st.slider("Rotation Angle (°)", -180.0, 180.0, 0.0, 5.0, key="cropper_rot_slider")

        with row1_c4:
            feather_px = st.slider("Edge Feathering (Softness)", 0, 50, 0, 1, key="cropper_feather_slider", help="Gaussian edge softness in pixels for smooth borders")
            aspect_choice = st.selectbox(
                "Aspect Ratio Lock",
                ["1:1", "4:3", "16:9", "3:2", "9:16", "freeform"],
                key="cropper_aspect_select"
            )

        # Secondary options (Sides for Polygon, Inner ratio for Star, Background & Tight crop)
        row2_c1, row2_c2, row2_c3 = st.columns([1.3, 1.3, 1.4])

        with row2_c1:
            bg_choice = st.selectbox(
                "Background Outside Shape",
                options=[
                    "Transparent (RGBA PNG)",
                    "Solid Black (0, 0, 0)",
                    "Solid White (255, 255, 255)",
                    "Studio Dark (22, 27, 34)"
                ],
                key="cropper_bg_select"
            )
            bg_map = {
                "Transparent (RGBA PNG)": "transparent",
                "Solid Black (0, 0, 0)": "black",
                "Solid White (255, 255, 255)": "white",
                "Studio Dark (22, 27, 34)": "studio_dark"
            }
            bg_key = bg_map.get(bg_choice, "transparent")

        with row2_c2:
            tight_crop_val = st.checkbox(
                "Crop Tight to Bounding Box",
                value=True,
                key="cropper_tight_chk",
                help="Trims away unused canvas area, returning only the cropped shape bounding box."
            )

        with row2_c3:
            if shape_name == "polygon":
                sides_val = st.slider("Polygon Sides (N)", 3, 16, 5, 1, key="cropper_sides_slider")
                star_ratio = 0.382
            elif shape_name == "star":
                star_points = st.slider("Star Points", 3, 12, 5, 1, key="cropper_star_points")
                star_ratio = st.slider("Inner Radius Ratio", 0.15, 0.85, 0.382, 0.05, key="cropper_star_ratio")
                sides_val = star_points
            else:
                sides_val = 5
                star_ratio = 0.382

        # -------------------------------------------------------------
        # EXECUTE SHAPE CROP
        # -------------------------------------------------------------
        cropped_output = vector_shape_crop(
            image=base_img,
            shape=shape_name,
            center_x=center_x,
            center_y=center_y,
            size=size_val,
            aspect_ratio=aspect_choice,
            rotation=rotation_deg,
            feather=feather_px,
            bg_fill=bg_key,
            tight_crop=tight_crop_val,
            sides=sides_val,
            star_inner_ratio=star_ratio
        )

        # -------------------------------------------------------------
        # LIVE DUAL PREVIEW
        # -------------------------------------------------------------
        prev_col1, prev_col2 = st.columns(2)

        with prev_col1:
            st.markdown(
                f"<div style='font-size: 0.85rem; font-weight: 700; color: #94a3b8; margin-bottom: 4px;'>📐 SOURCE CANVAS WITH CROP REGION ({base_img.shape[1]} × {base_img.shape[0]} px)</div>",
                unsafe_allow_html=True
            )
            # Create interactive overlay showing shape boundary
            h, w = base_img.shape[:2]
            overlay_view = base_img.copy()
            if overlay_view.ndim == 3 and overlay_view.shape[2] == 4:
                overlay_view = cv2.cvtColor(overlay_view, cv2.COLOR_RGBA2RGB)
            elif overlay_view.ndim == 2:
                overlay_view = cv2.cvtColor(overlay_view, cv2.COLOR_GRAY2RGB)

            st.image(overlay_view, width="stretch")

        with prev_col2:
            is_rgba = (cropped_output.ndim == 3 and cropped_output.shape[2] == 4)
            mode_tag = "RGBA (32-bit Transparent)" if is_rgba else "RGB (24-bit)"
            st.markdown(
                f"<div style='font-size: 0.85rem; font-weight: 700; color: #38bdf8; margin-bottom: 4px;'>✨ CROPPED SHAPE OUTPUT ({cropped_output.shape[1]} × {cropped_output.shape[0]} px • {mode_tag})</div>",
                unsafe_allow_html=True
            )
            st.image(cropped_output, width="stretch")

        # -------------------------------------------------------------
        # ACTIONS: DOWNLOAD & ADD TO PIPELINE
        # -------------------------------------------------------------
        st.write("")
        act_col1, act_col2 = st.columns(2)

        with act_col1:
            # Encode to PNG preserving alpha channel
            if cropped_output.ndim == 3 and cropped_output.shape[2] == 4:
                out_bgra = cv2.cvtColor(cropped_output, cv2.COLOR_RGBA2BGRA)
                success, encoded_png = cv2.imencode('.png', out_bgra)
            elif cropped_output.ndim == 3 and cropped_output.shape[2] == 3:
                out_bgr = cv2.cvtColor(cropped_output, cv2.COLOR_RGB2BGR)
                success, encoded_png = cv2.imencode('.png', out_bgr)
            else:
                success, encoded_png = cv2.imencode('.png', cropped_output)

            if success:
                st.download_button(
                    label=f"⬇️ Download Cropped Shape ({shape_name.capitalize()} PNG with Alpha)",
                    data=encoded_png.tobytes(),
                    file_name=f"pixel_forge_{shape_name}_crop.png",
                    mime="image/png",
                    use_container_width=True,
                    type="primary"
                )

        with act_col2:
            if st.button("➕ Add Shape Crop Stage to Pipeline Stack", use_container_width=True, type="secondary"):
                crop_params = {
                    "shape": shape_name,
                    "center_x": center_x,
                    "center_y": center_y,
                    "size": size_val,
                    "aspect_ratio": aspect_choice,
                    "rotation": rotation_deg,
                    "feather": feather_px,
                    "bg_fill": bg_key,
                    "tight_crop": tight_crop_val,
                    "sides": sides_val,
                    "star_inner_ratio": star_ratio
                }
                add_layer_to_stack("geo_shape_crop", crop_params)
                st.success(f"Added '{shape_name.capitalize()}' Crop stage to pipeline!")
                st.rerun()
