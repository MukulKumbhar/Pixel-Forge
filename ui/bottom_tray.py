"""
ImageLab Pro - Unified Resizable & Collapsible Studio Bottom Tray
Houses all filter controls: Active Stage Parameters, Filter Catalog (66 tools across 7 Domains),
Comparative Histograms, and Export Studio inside an independently scrollable,
manually resizable (stretch/shrink), and collapsible dock.
"""

import io
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from algorithms.registry import ALGORITHM_REGISTRY, DOMAINS
from core.metrics import compute_histograms
from core.state_manager import add_layer_to_stack
from ui.parameter_drawer import render_parameter_drawer


def render_bottom_tray(raw_img: np.ndarray, proc_img: np.ndarray):
    """
    Renders the unified studio bottom tray:
    1. Minimized floating dock bar when collapsed (1-click expand to restore).
    2. Header bar when expanded (title, active stage indicator, manual height stretch/shrink, 1-click hide button).
    3. Resizable scrollable container with 4 dedicated tabs:
       - ⚙️ Active Filter Settings (Parameter Drawer & Micro-Info)
       - 🗂️ Browse Filter Catalog (66 Tools across 7 Domains)
       - 📊 Comparative Histograms (Luminance & RGB Scopes)
       - 💾 Export Studio (PNG & Recipe JSON)
    """
    if "tray_collapsed" not in st.session_state:
        st.session_state["tray_collapsed"] = False

    if "tray_height" not in st.session_state:
        st.session_state["tray_height"] = 580

    # -------------------------------------------------------------
    # 1. COLLAPSED / MINIMIZED STATE
    # -------------------------------------------------------------
    if st.session_state["tray_collapsed"]:
        min_col1, min_col2 = st.columns([4, 1.4])
        with min_col1:
            st.markdown(
                """
                <div class="tray-minimized-bar">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 1.3rem;">🎛️</span>
                        <div>
                            <span style="font-size: 0.98rem; font-weight: 700; color: #f8fafc;">Filter & Parameter Studio Tray (Minimized)</span>
                            <span style="font-size: 0.82rem; color: #94a3b8; margin-left: 10px;">Full screen canvas preview active</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with min_col2:
            st.write("")
            if st.button("🔼 Show Filter Tray", use_container_width=True, type="primary"):
                st.session_state["tray_collapsed"] = False
                st.rerun()
        return

    # -------------------------------------------------------------
    # 2. EXPANDED TRAY: HEADER CONTROLS
    # -------------------------------------------------------------
    h_col1, h_col2, h_col3 = st.columns([1.8, 2.0, 1.1])
    with h_col1:
        selected_id = st.session_state.get("selected_layer_id")
        if selected_id:
            layer_name = next(
                (l["name"] for l in st.session_state.get("active_layer_stack", []) if l["id"] == selected_id),
                "Unknown"
            )
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; gap: 10px; margin-top: 6px;">
                    <span style="font-size: 1.25rem;">🎛️</span>
                    <span style="font-size: 1.05rem; font-weight: 700; color: #f8fafc;">Studio Filter Tray</span>
                    <span class="badge badge-rgb">Editing: {layer_name}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="display: flex; align-items: center; gap: 10px; margin-top: 6px;">
                    <span style="font-size: 1.25rem;">🎛️</span>
                    <span style="font-size: 1.05rem; font-weight: 700; color: #f8fafc;">Studio Filter Tray</span>
                    <span class="badge badge-unit">66 Algorithms Ready</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    with h_col2:
        # Tray Height Presets (Stretch / Shrink)
        cur_h = st.session_state.get("tray_height", 580)
        idx = 1
        if cur_h <= 480:
            idx = 0
        elif cur_h >= 680:
            idx = 2

        size_choice = st.radio(
            "Tray Height Scale",
            options=["Standard (460px)", "Spacious Studio (580px)", "Full Workstation (720px)"],
            index=idx,
            horizontal=True,
            key="tray_size_preset_radio",
            label_visibility="collapsed"
        )
        if size_choice == "Standard (460px)" and cur_h != 460:
            st.session_state["tray_height"] = 460
            st.rerun()
        elif size_choice == "Spacious Studio (580px)" and cur_h != 580:
            st.session_state["tray_height"] = 580
            st.rerun()
        elif size_choice == "Full Workstation (720px)" and cur_h != 720:
            st.session_state["tray_height"] = 720
            st.rerun()

    with h_col3:
        if st.button("🔽 Hide Tray", key="btn_hide_tray_header", use_container_width=True, help="Collapse tray to view image preview in full screen"):
            st.session_state["tray_collapsed"] = True
            st.rerun()

    # -------------------------------------------------------------
    # 3. RESIZABLE & INDEPENDENTLY SCROLLABLE CONTAINER
    # -------------------------------------------------------------
    tray_h = st.session_state.get("tray_height", 580)

    # Wrap in custom CSS resizable container with independent internal scroll
    st.markdown(
        f'<div class="studio-bottom-tray" style="height: {tray_h}px;"><div class="tray-scroll-body" style="height: {tray_h - 15}px;">',
        unsafe_allow_html=True
    )

    tab_labels = [
        "⚙️ Active Filter Settings",
        "🗂️ Browse Filter Catalog (66 Tools)",
        "📊 Comparative Histograms",
        "💾 Export Studio & Recipe"
    ]
    tray_tabs = st.tabs(tab_labels)

    # --- TAB 1: ACTIVE FILTER SETTINGS ---
    with tray_tabs[0]:
        selected_layer_id = st.session_state.get("selected_layer_id")
        render_parameter_drawer(selected_layer_id)

    # --- TAB 2: BROWSE FILTER CATALOG (66 TOOLS ACROSS 7 DOMAINS) ---
    with tray_tabs[1]:
        _render_catalog_shelf()

    # --- TAB 3: COMPARATIVE HISTOGRAMS ---
    with tray_tabs[2]:
        _render_histograms(raw_img, proc_img)

    # --- TAB 4: EXPORT STUDIO & RECIPE ---
    with tray_tabs[3]:
        _render_export_studio(proc_img)

    st.markdown('</div></div>', unsafe_allow_html=True)


def _render_catalog_shelf():
    """Renders the 7 domain tabs and 66 tool cards inside the catalog shelf."""
    domain_keys = list(DOMAINS.keys())
    domain_titles = [DOMAINS[k]["title"] for k in domain_keys]

    d_tabs = st.tabs(domain_titles)

    for i, tab in enumerate(d_tabs):
        d_key = domain_keys[i]
        d_info = DOMAINS[d_key]

        with tab:
            st.markdown(
                f"<div style='font-size: 0.88rem; color: #94a3b8; margin-bottom: 0.9rem; line-height: 1.5;'>{d_info['description']}</div>",
                unsafe_allow_html=True
            )

            # Get algorithms in this domain
            domain_algos = [
                (k, v) for k, v in ALGORITHM_REGISTRY.items() if v.get("domain") == d_key
            ]

            # Render in 3-column grid
            cols = st.columns(3)
            for idx, (algo_key, meta) in enumerate(domain_algos):
                col = cols[idx % 3]
                with col:
                    in_contract = meta.get("input_contract", "ANY")
                    badge_class = "badge-rgb" if in_contract == "RGB" else ("badge-binary" if in_contract == "BINARY" else "badge-gray")
                    summary = meta.get("summary", "")

                    with st.container():
                        st.markdown(
                            f"""
                            <div class="dock-card">
                                <div>
                                    <div class="dock-card-title" title="{meta['name']}">
                                        {meta['name']}
                                    </div>
                                    <div class="dock-card-desc">
                                        {summary}
                                    </div>
                                </div>
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                                    <span class="badge {badge_class}">Requires: {in_contract}</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        if st.button(f"➕ Add to Pipeline", key=f"add_{algo_key}", use_container_width=True):
                            add_layer_to_stack(algo_key)
                            st.rerun()


def _render_histograms(raw_img: np.ndarray, proc_img: np.ndarray):
    """Renders comparative histogram distributions inside Tab 3."""
    hist_mode = st.radio(
        "Channel Selection",
        ["Luminance / Intensity", "RGB Multi-Channel"],
        horizontal=True,
        key="tray_hist_channel_radio"
    )

    hist_in = compute_histograms(raw_img)
    hist_out = compute_histograms(proc_img)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.4), facecolor='#0e1117')
    for ax in (ax1, ax2):
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#9ca3af', labelsize=9)
        ax.grid(color='#374151', linestyle='--', linewidth=0.5, alpha=0.6)
        for spine in ax.spines.values():
            spine.set_color('#374151')

    x = np.arange(256)

    if hist_mode == "RGB Multi-Channel" and hist_in["mode"] == "rgb" and hist_out["mode"] == "rgb":
        ax1.plot(x, hist_in["r"], color='#ef4444', label='Red', alpha=0.85, linewidth=1.5)
        ax1.plot(x, hist_in["g"], color='#22c55e', label='Green', alpha=0.85, linewidth=1.5)
        ax1.plot(x, hist_in["b"], color='#3b82f6', label='Blue', alpha=0.85, linewidth=1.5)

        ax2.plot(x, hist_out["r"], color='#ef4444', label='Red', alpha=0.85, linewidth=1.5)
        ax2.plot(x, hist_out["g"], color='#22c55e', label='Green', alpha=0.85, linewidth=1.5)
        ax2.plot(x, hist_out["b"], color='#3b82f6', label='Blue', alpha=0.85, linewidth=1.5)
        ax1.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0', fontsize=8)
        ax2.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0', fontsize=8)
    else:
        ax1.fill_between(x, hist_in["luminance"], color='#818cf8', alpha=0.45)
        ax1.plot(x, hist_in["luminance"], color='#a5b4fc', linewidth=1.5)

        ax2.fill_between(x, hist_out["luminance"], color='#34d399', alpha=0.45)
        ax2.plot(x, hist_out["luminance"], color='#6ee7b7', linewidth=1.5)

    ax1.set_title("Input Frame Tonal Distribution", color='#f8fafc', fontsize=10, pad=8, fontweight='bold')
    ax2.set_title("Processed Output Tonal Distribution", color='#f8fafc', fontsize=10, pad=8, fontweight='bold')
    ax1.set_xlim([0, 255])
    ax2.set_xlim([0, 255])

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _render_export_studio(proc_img: np.ndarray):
    """Renders export and recipe download/upload controls inside Tab 4."""
    st.markdown(
        """
        <div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.6rem;">
            💾 Export Final Master & Share Pipeline Recipe
        </div>
        """,
        unsafe_allow_html=True
    )
    exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 1.2])

    with exp_col1:
        pil_out = Image.fromarray(proc_img)
        buf_png = io.BytesIO()
        pil_out.save(buf_png, format="PNG")
        st.download_button(
            label="💾 Download Master PNG",
            data=buf_png.getvalue(),
            file_name="imagelab_processed.png",
            mime="image/png",
            use_container_width=True
        )

    with exp_col2:
        recipe = {
            "version": "1.0",
            "layers": st.session_state.get("active_layer_stack", [])
        }
        recipe_json = json.dumps(recipe, indent=2)
        st.download_button(
            label="📋 Download Recipe JSON",
            data=recipe_json,
            file_name="imagelab_recipe.json",
            mime="application/json",
            use_container_width=True
        )

    with exp_col3:
        recipe_file = st.file_uploader("Import Recipe JSON", type=["json"], label_visibility="collapsed")
        if recipe_file is not None:
            try:
                data = json.load(recipe_file)
                if "layers" in data:
                    st.session_state["active_layer_stack"] = data["layers"]
                    st.success("Pipeline recipe imported successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to parse recipe: {e}")
