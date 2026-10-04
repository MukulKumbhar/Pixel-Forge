"""
Pixel-Forge — Interactive Digital Image Processing Studio
Entry Point: Orchestrates layout, state, sidebar pipeline stack, and dual viewport rendering.
"""

import io
import json
import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Core imports
from core.pipeline import execute_pipeline
from core.state_manager import (
    PRESETS,
    clear_layer_stack,
    init_session_state,
    load_preset,
    move_layer_down,
    move_layer_up,
    redo,
    remove_layer,
    toggle_layer_autoconvert,
    toggle_layer_enabled,
    undo,
)
from algorithms.registry import ALGORITHM_REGISTRY, DOMAINS
from ui.styles import apply_studio_styles
from ui.viewport import (
    render_dual_viewport,
    render_telemetry_banner,
)
from ui.bottom_tray import render_bottom_tray
from ui.conflict_resolver import render_conflict_alerts

# Set page config
st.set_page_config(
    page_title="Pixel-Forge — Interactive DIP Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom dark-mode DCC studio CSS styling
apply_studio_styles()

# Initialize session state variables
init_session_state()


def load_image_asset(filename: str) -> np.ndarray:
    """Loads a sample benchmark image from the assets directory."""
    path = os.path.join("assets", "sample_images", filename)
    if os.path.exists(path):
        bgr = cv2.imread(path)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # Fallback synthetic pattern if file is missing
    return np.zeros((512, 512, 3), dtype=np.uint8) + 128


def main():
    # Guarantee session state initialization for this user thread/session
    init_session_state()

    # -------------------------------------------------------------
    # 1. STUDIO HEADER BAR
    # -------------------------------------------------------------
    st.markdown(
        """
        <div class="studio-header">
            <div class="studio-title-box">
                <span style="font-size: 1.8rem;">🔬</span>
                <div>
                    <h1 class="studio-title">Pixel-Forge</h1>
                    <p class="studio-subtitle">Professional Digital Image Processing Studio & Dynamic Pipeline Workbench</p>
                </div>
            </div>
            <div style="text-align: right;">
                <span class="badge badge-unit">Academic Units 1 — 6 + Photoshop Studio</span>
                <span class="badge badge-rgb" style="margin-left: 6px;">66 Real-Time Algorithms</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------------------------------------
    # 2. IMAGE LOADER & TOP WORKSPACE BAR
    # -------------------------------------------------------------
    top_col1, top_col2, top_col3 = st.columns([1.5, 1.5, 1])

    with top_col1:
        sample_options = {
            "chroma_studio.png": "🎨 Natural Scene & Color Spectrum (Colorimetry/Radiometric)",
            "resolution_target.png": "🎯 Siemens Star & Frequency Wedges (Geometry/Sampling)",
            "document_ocr.png": "📄 Degraded Scanned Document (Otsu/Sharpen/Morphology)",
            "geometry_shapes.png": "📐 Geometric Shapes & Vectors (Hough/Canny/Sobel)"
        }
        selected_sample = st.selectbox(
            "Select Benchmark Sample",
            options=list(sample_options.keys()),
            format_func=lambda k: sample_options[k],
            key="sample_selector"
        )

    with top_col2:
        uploaded_file = st.file_uploader(
            "Or Upload Custom Image",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            help="Upload your own photographic or medical image to test through the pipeline."
        )

    # Resolve active base frame
    if uploaded_file is not None:
        try:
            pil_img = Image.open(uploaded_file).convert("RGB")
            raw_base_image = np.array(pil_img, dtype=np.uint8)
        except Exception as e:
            st.error(f"Error loading uploaded image: {e}")
            raw_base_image = load_image_asset(selected_sample)
    else:
        raw_base_image = load_image_asset(selected_sample)

    with top_col3:
        st.write("")  # Spacing
        undo_col, redo_col, reset_col = st.columns(3)
        with undo_col:
            can_undo = len(st.session_state.get("history_stack", [])) > 0
            if st.button("↩ Undo", disabled=not can_undo, use_container_width=True):
                undo()
                st.rerun()
        with redo_col:
            can_redo = len(st.session_state.get("future_stack", [])) > 0
            if st.button("↪ Redo", disabled=not can_redo, use_container_width=True):
                redo()
                st.rerun()
        with reset_col:
            if st.button("🗑️ Reset", use_container_width=True, help="Clear active layer stack"):
                clear_layer_stack()
                st.rerun()

    # -------------------------------------------------------------
    # 3. SIDEBAR: ACTIVE STACK & PRESET MANAGER
    # -------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🎞️ **Active Layer Stack**")
        st.caption("Non-destructive serial chain evaluated in milliseconds.")

        # Preset selector callback to eliminate infinite rerun loop
        def on_preset_dropdown_change():
            chosen = st.session_state.get("preset_dropdown_choice")
            if chosen and chosen != "-- Select Preset --":
                load_preset(chosen)
                # Reset selection back to default so subsequent reruns never loop!
                st.session_state["preset_dropdown_choice"] = "-- Select Preset --"

        st.selectbox(
            "Load Workflow Preset",
            options=["-- Select Preset --"] + list(PRESETS.keys()),
            format_func=lambda k: PRESETS[k]["title"] if k in PRESETS else k,
            key="preset_dropdown_choice",
            on_change=on_preset_dropdown_change
        )

        # 1-Click Quick Preset Buttons
        st.markdown("<div style='font-size: 0.84rem; font-weight: 700; color: #cbd5e1; margin-top: 10px; margin-bottom: 6px;'>⚡ QUICK PRESETS:</div>", unsafe_allow_html=True)
        qp_col1, qp_col2 = st.columns(2)
        with qp_col1:
            if st.button("✨ Retouch", key="qp_retouch", use_container_width=True, help="Glamour skin smoothing + vibrance + unsharp mask"):
                load_preset("photoshop_portrait")
                st.rerun()
            if st.button("🎬 Film Grade", key="qp_film", use_container_width=True):
                load_preset("film_grade")
                st.rerun()
            if st.button("📐 Edge Hough", key="qp_edge", use_container_width=True):
                load_preset("edge_hough")
                st.rerun()
        with qp_col2:
            if st.button("🎨 Sketch", key="qp_sketch", use_container_width=True, help="Graphite pencil drawing + optical vignette"):
                load_preset("artistic_sketch")
                st.rerun()
            if st.button("📄 Doc Clean", key="qp_doc", use_container_width=True):
                load_preset("doc_cleaning")
                st.rerun()
            if st.button("🔬 Denoise", key="qp_denoise", use_container_width=True):
                load_preset("noise_restoration")
                st.rerun()

        st.divider()

        # Render active layers
        stack = st.session_state.active_layer_stack

        if not stack:
            st.info("Stack is empty. Select a tool from the **Bottom Filter Catalog** or load a **Workflow Preset** above.")
        else:
            for idx, layer in enumerate(stack):
                layer_id = layer["id"]
                is_selected = (st.session_state.selected_layer_id == layer_id)
                is_enabled = layer.get("enabled", True)
                algo_meta = ALGORITHM_REGISTRY.get(layer.get("algorithm_key"), {})
                in_c = algo_meta.get("input_contract", "ANY")

                card_class = "layer-card"
                if is_selected:
                    card_class += " active-layer"
                if not is_enabled:
                    card_class += " bypassed"

                with st.container():
                    st.markdown(
                        f"""
                        <div class="{card_class}">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                <strong style="font-size: 0.94rem; color: #f8fafc;">
                                    {idx + 1}. {layer['name']}
                                </strong>
                                <span class="badge badge-unit">{in_c}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Controls row: [Toggle] [Up] [Down] [Gear/Edit] [Delete]
                    c_toggle, c_up, c_down, c_gear, c_del = st.columns([1, 1, 1, 1, 1])

                    with c_toggle:
                        chk = st.checkbox("Active", value=is_enabled, key=f"layer_chk_{layer_id}", label_visibility="collapsed")
                        if chk != is_enabled:
                            toggle_layer_enabled(layer_id)
                            st.rerun()

                    with c_up:
                        if st.button("▲", key=f"btn_up_{layer_id}", disabled=(idx == 0)):
                            move_layer_up(idx)
                            st.rerun()

                    with c_down:
                        if st.button("▼", key=f"btn_down_{layer_id}", disabled=(idx == len(stack) - 1)):
                            move_layer_down(idx)
                            st.rerun()

                    with c_gear:
                        btn_label = "⚙️" if not is_selected else "★"
                        if st.button(btn_label, key=f"btn_gear_{layer_id}", help="Open parameter drawer"):
                            st.session_state.selected_layer_id = layer_id if not is_selected else None
                            st.rerun()

                    with c_del:
                        if st.button("✕", key=f"btn_del_{layer_id}", help="Delete stage"):
                            remove_layer(layer_id)
                            st.rerun()

            st.divider()
            # Clear stack button
            if st.button("Clear All Stages", use_container_width=True, type="secondary"):
                clear_layer_stack()
                st.rerun()

    # -------------------------------------------------------------
    # 4. EXECUTE PIPELINE
    # -------------------------------------------------------------
    pipeline_result = execute_pipeline(raw_base_image, st.session_state.active_layer_stack)
    processed_image = pipeline_result.output_image

    # Prerequisite & Conflict resolver alerts
    render_conflict_alerts(pipeline_result.stage_results)

    # Telemetry diagnostics banner
    render_telemetry_banner(raw_base_image, processed_image, pipeline_result)

    # Fixed Sticky Dual-Viewport
    render_dual_viewport(raw_base_image, processed_image)

    # -------------------------------------------------------------
    # 5. UNIFIED RESIZABLE & COLLAPSIBLE BOTTOM FILTER TRAY
    # -------------------------------------------------------------
    render_bottom_tray(raw_base_image, processed_image)


if __name__ == "__main__":
    main()
