"""
Pixel-Forge — Interactive Digital Image Processing Studio
Entry Point: Orchestrates layout, state, sidebar pipeline stack, and responsive viewport rendering.
"""

import io
import json
import os
import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Eliminate PIL decompression bomb cap for large 200MB+ images
import streamlit as st

# Core imports
from core.pipeline import execute_pipeline
from core.state_manager import (
    PRESETS,
    add_layer_to_stack,
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
    render_histogram_dock,
    render_telemetry_banner,
)
from ui.parameter_drawer import render_parameter_drawer, render_filter_catalog
from ui.conflict_resolver import render_conflict_alerts
from ui.compressor_dock import render_compressor_dock
from ui.cropper_dock import render_cropper_dock

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
                <span class="badge badge-unit">Academic Units 1 — 8 + Photoshop Studio</span>
                <span class="badge badge-rgb" style="margin-left: 6px;">70 Real-Time Algorithms</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------------------------------------
    # 2. IMAGE LOADER & TOP WORKSPACE BAR
    # -------------------------------------------------------------
    top_col1, top_col2, top_col3 = st.columns([1.6, 1.6, 1.2])

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
            "Or Upload Custom Image (Supports up to 1,000 MB)",
            type=["png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"],
            help="High-capacity upload buffer supports multi-megabyte photographic, medical, and RAW scans (up to 1,000 MB)."
        )

    # High-Performance Cached Ingestion for Large Images
    if uploaded_file is not None:
        file_sig = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("_cached_upload_sig") == file_sig and st.session_state.get("_master_raw_image") is not None:
            master_raw_image = st.session_state["_master_raw_image"]
        else:
            with st.spinner(f"⚡ Ingesting & caching high-resolution master image ({uploaded_file.size / (1024*1024):.1f} MB)..."):
                try:
                    Image.MAX_IMAGE_PIXELS = None
                    file_bytes = uploaded_file.getvalue()
                    pil_img = Image.open(io.BytesIO(file_bytes))
                    if pil_img.mode in ("RGBA", "LA"):
                        master_raw_image = np.array(pil_img.convert("RGBA"), dtype=np.uint8)
                    else:
                        master_raw_image = np.array(pil_img.convert("RGB"), dtype=np.uint8)

                    st.session_state["_cached_upload_sig"] = file_sig
                    st.session_state["_master_raw_image"] = master_raw_image
                    st.session_state["_proxy_raw_image"] = None  # Invalidate proxy cache
                    import gc
                    gc.collect()
                except Exception as e:
                    st.error(f"Error loading uploaded image: {e}")
                    master_raw_image = load_image_asset(selected_sample)
    else:
        master_raw_image = load_image_asset(selected_sample)
        if st.session_state.get("_cached_upload_sig") not in [None, "sample"]:
            st.session_state["_cached_upload_sig"] = "sample"
            st.session_state["_master_raw_image"] = master_raw_image
            st.session_state["_proxy_raw_image"] = None

    # Studio Viewport Engine: Resolve active working frame
    master_h, master_w = master_raw_image.shape[:2]
    is_ultra_high_res = max(master_h, master_w) > 2048

    if is_ultra_high_res:
        if st.session_state.get("_proxy_raw_image") is None or st.session_state.get("_proxy_raw_image").shape[0] == 0:
            scale_factor = 2048.0 / max(master_h, master_w)
            target_pw = max(16, int(master_w * scale_factor))
            target_ph = max(16, int(master_h * scale_factor))
            proxy_img = cv2.resize(master_raw_image, (target_pw, target_ph), interpolation=cv2.INTER_AREA)
            st.session_state["_proxy_raw_image"] = proxy_img
        raw_base_image = st.session_state["_proxy_raw_image"]

        st.markdown(
            f"""
            <div style="background: rgba(99, 102, 241, 0.12); border: 1px solid rgba(99, 102, 241, 0.35); border-radius: 8px; padding: 6px 14px; margin-top: 8px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="color: #818cf8; font-weight: 700; font-size: 0.85rem;">⚡ Studio High-Performance Engine:</span>
                    <span style="color: #cbd5e1; font-size: 0.82rem; margin-left: 6px;">
                        Master Buffer: <b>{master_w} × {master_h} px</b> | Interactive Viewport: <b>{raw_base_image.shape[1]} × {raw_base_image.shape[0]} px</b>
                    </span>
                </div>
                <span class="badge badge-unit">100% Full-Resolution Preserved on Export</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        raw_base_image = master_raw_image

    with top_col3:
        st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #94a3b8; margin-bottom: 7px;'>PIPELINE HISTORY</div>", unsafe_allow_html=True)
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
    # 3. SIDEBAR: PIPELINE STACK & REORDERING
    # -------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🎞️ **Pipeline Layer Stack**")
        st.caption("Non-destructive serial chain evaluated in milliseconds.")

        # Preset selector callback to eliminate infinite rerun loop
        def on_preset_dropdown_change():
            chosen = st.session_state.get("preset_dropdown_choice")
            if chosen and chosen != "-- Select Preset --":
                load_preset(chosen)
                if st.session_state.get("active_layer_stack"):
                    first_id = st.session_state.active_layer_stack[0]["id"]
                    st.session_state.selected_layer_id = first_id
                    st.session_state["param_active_stage_select"] = first_id
                st.session_state["preset_dropdown_choice"] = "-- Select Preset --"

        st.selectbox(
            "Load Workflow Preset",
            options=["-- Select Preset --"] + list(PRESETS.keys()),
            format_func=lambda k: PRESETS[k]["title"] if k in PRESETS else k,
            key="preset_dropdown_choice",
            on_change=on_preset_dropdown_change
        )

        st.divider()

        # Render active layers list
        stack = st.session_state.active_layer_stack

        def _on_toggle_layer_cb(lid: str, w_key: str):
            new_state = st.session_state.get(w_key, True)
            for l in st.session_state.get("active_layer_stack", []):
                if l["id"] == lid:
                    l["enabled"] = new_state
                    break

        if not stack:
            st.info("Pipeline is empty. Open the **Filter Catalog** expander in the workspace or choose a preset above to begin.")
        else:
            st.markdown(f"<div style='font-size: 0.84rem; font-weight: 700; color: #cbd5e1; margin-bottom: 6px;'>ACTIVE STAGES ({len(stack)}):</div>", unsafe_allow_html=True)
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
                                <strong style="font-size: 0.92rem; color: #f8fafc;">
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
                        chk_key = f"layer_chk_{layer_id}"
                        if chk_key not in st.session_state:
                            st.session_state[chk_key] = is_enabled
                        st.checkbox("Active", key=chk_key, label_visibility="collapsed", on_change=_on_toggle_layer_cb, args=(layer_id, chk_key))

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
                        if st.button(btn_label, key=f"btn_gear_{layer_id}", help="Tweak stage parameters"):
                            target_id = layer_id if not is_selected else None
                            st.session_state.selected_layer_id = target_id
                            if target_id:
                                st.session_state["param_active_stage_select"] = target_id
                            st.rerun()

                    with c_del:
                        if st.button("✕", key=f"btn_del_{layer_id}", help="Delete stage"):
                            remove_layer(layer_id)
                            if st.session_state.get("selected_layer_id") == layer_id:
                                st.session_state.selected_layer_id = None
                                st.session_state.pop("param_active_stage_select", None)
                            st.rerun()

            st.write("")
            if st.button("Clear All Stages", use_container_width=True, type="secondary"):
                clear_layer_stack()
                st.session_state.pop("param_active_stage_select", None)
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

    # Full-Screen Responsive Dual-Viewport Canvas
    render_dual_viewport(raw_base_image, processed_image)

    # -------------------------------------------------------------
    # 5. STUDIO CONTROL EXPANDERS SUITE
    # -------------------------------------------------------------
    # Expander 1: Filter Catalog & Algorithm Selector (Browse & Add Stages)
    with st.expander("🎛️ **Filter Catalog & Algorithm Selector (Browse & Add Stages)**", expanded=False):
        render_filter_catalog()

    # Expander 2: Active Filter Parameters & Stage Controls
    has_active_layers = len(st.session_state.get("active_layer_stack", [])) > 0
    with st.expander("⚙️ **Active Filter Parameters & Stage Controls**", expanded=has_active_layers):
        render_parameter_drawer(st.session_state.get("selected_layer_id"))

    # Expander 3: Vector Multi-Shape Cropper Studio
    render_cropper_dock(raw_base_image, processed_image, master_raw_image)

    # Expander 4: Pro Image Compressor (Lossless & Deep Lossy with Quality Slider)
    render_compressor_dock(raw_base_image, processed_image, master_raw_image)

    # Expander 5: Comparative Image Histograms (Input vs Output)
    render_histogram_dock(raw_base_image, processed_image)

    # Expander 6: Export Studio & Pipeline Recipe
    with st.expander("💾 **Export Studio & Pipeline Recipe**", expanded=False):
        exp_col1, exp_col2, exp_col3 = st.columns([1.2, 1.2, 1.4])

        with exp_col1:
            st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px;'>📸 Export Processed Image</div>", unsafe_allow_html=True)
            export_mode = st.radio(
                "Export Quality Mode",
                options=["Full-Resolution Master (100% Original Quality)", "Active Viewport Canvas"],
                horizontal=True,
                key="export_res_choice"
            )

            # Determine frame to export
            if export_mode == "Full-Resolution Master (100% Original Quality)" and is_ultra_high_res:
                if len(st.session_state.get("active_layer_stack", [])) > 0:
                    with st.spinner(f"⚡ Rendering full-resolution master ({master_w} × {master_h} px) through active pipeline..."):
                        master_res = execute_pipeline(master_raw_image, st.session_state.active_layer_stack)
                        final_export = master_res.output_image
                else:
                    final_export = master_raw_image
            else:
                final_export = processed_image

            is_rgba = (final_export.ndim == 3 and final_export.shape[2] == 4)
            is_rgb = (final_export.ndim == 3 and final_export.shape[2] == 3)
            if is_rgba:
                out_bgr = cv2.cvtColor(final_export, cv2.COLOR_RGBA2BGRA)
            elif is_rgb:
                out_bgr = cv2.cvtColor(final_export, cv2.COLOR_RGB2BGR)
            else:
                out_bgr = final_export

            success, encoded_img = cv2.imencode('.png', out_bgr)
            if success:
                st.download_button(
                    label=f"⬇️ Download Full-Res PNG ({final_export.shape[1]} × {final_export.shape[0]} px)",
                    data=encoded_img.tobytes(),
                    file_name="pixel_forge_export.png",
                    mime="image/png",
                    use_container_width=True,
                    type="primary"
                )

        with exp_col2:
            st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px;'>📋 Export Pipeline Recipe</div>", unsafe_allow_html=True)
            recipe = {
                "studio": "Pixel-Forge",
                "version": "1.0",
                "layers": st.session_state.active_layer_stack
            }
            recipe_json = json.dumps(recipe, indent=2)
            st.download_button(
                label="⬇️ Download Recipe JSON",
                data=recipe_json,
                file_name="pixel_forge_recipe.json",
                mime="application/json",
                use_container_width=True
            )

        with exp_col3:
            st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px;'>📂 Import Pipeline Recipe</div>", unsafe_allow_html=True)
            recipe_file = st.file_uploader("Upload Recipe JSON", type=["json"], label_visibility="collapsed")
            if recipe_file is not None:
                try:
                    data = json.load(recipe_file)
                    if "layers" in data:
                        st.session_state["active_layer_stack"] = data["layers"]
                        st.success("Pipeline recipe imported successfully!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to parse recipe: {e}")


if __name__ == "__main__":
    main()
