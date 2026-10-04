"""
Pixel-Forge - Dual-Pane Viewport & Diagnostics Studio
Renders full-screen high-resolution side-by-side comparison, real-time radiometric telemetry,
and synchronized comparative histogram distributions.
"""

from typing import Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from algorithms.registry import ALGORITHM_REGISTRY, DOMAINS
from core.metrics import (
    calculate_entropy,
    calculate_mse,
    calculate_psnr,
    calculate_ssim,
    compute_histograms,
)
from core.pipeline import PipelineExecutionResult, detect_state


def render_telemetry_banner(raw_img: np.ndarray, proc_img: np.ndarray, result: PipelineExecutionResult):
    """
    Renders real-time telemetry diagnostics banner:
    Execution Time, Active Layers, Dynamic Range, Color Space, PSNR, MSE, SSIM, Entropy.
    """
    active_count = sum(1 for r in result.stage_results if r.status in ["executed", "converted"])
    total_count = len(result.stage_results)

    fmt_name = detect_state(proc_img)
    if fmt_name == "RGB":
        color_label = "RGB (24-bit)"
    elif fmt_name == "BINARY":
        color_label = "Binary (1-bit)"
    else:
        color_label = "Grayscale (8-bit)"

    dr_min = int(np.min(proc_img))
    dr_max = int(np.max(proc_img))

    psnr_val = calculate_psnr(raw_img, proc_img)
    mse_val = calculate_mse(raw_img, proc_img)
    ssim_val = calculate_ssim(raw_img, proc_img)
    entropy_val = calculate_entropy(proc_img)

    st.markdown(
        f"""
        <div class="telemetry-grid">
            <div class="telemetry-card">
                <div class="telemetry-label">⏱️ Latency</div>
                <div class="telemetry-value" style="color: #38bdf8;">{result.total_time_ms:.1f} ms</div>
            </div>
            <div class="telemetry-card">
                <div class="telemetry-label">🎞️ Active Layers</div>
                <div class="telemetry-value" style="color: #a78bfa;">{active_count} / {total_count}</div>
            </div>
            <div class="telemetry-card">
                <div class="telemetry-label">🎨 Color Space</div>
                <div class="telemetry-value" style="color: #34d399;">{color_label}</div>
            </div>
            <div class="telemetry-card">
                <div class="telemetry-label">📊 Dynamic Range</div>
                <div class="telemetry-value">[{dr_min} — {dr_max}]</div>
            </div>
            <div class="telemetry-card">
                <div class="telemetry-label">📈 PSNR</div>
                <div class="telemetry-value" style="color: #fbbf24;">{psnr_val:.2f} dB</div>
            </div>
            <div class="telemetry-card">
                <div class="telemetry-label">🔍 SSIM Index</div>
                <div class="telemetry-value" style="color: #f472b6;">{ssim_val:.3f}</div>
            </div>
            <div class="telemetry-card">
                <div class="telemetry-label">📉 Entropy</div>
                <div class="telemetry-value">{entropy_val:.2f} b/p</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_dual_viewport(raw_img: np.ndarray, proc_img: np.ndarray):
    """
    Renders the responsive full-screen dual-viewport canvas.
    Modes:
      1. Side-by-Side (Dual): Direct comparison of source frame and processed output.
      2. Full Canvas (Processed): Clean, unobstructed view of processed output.
      3. Interactive Split Wipe: Dynamic horizontal reveal slider between input and output.
      4. Difference Heatmap: Residual error magnitude (|Source - Processed|) via thermal JET colormap.
    """
    with st.container():
        ctrl_col1, ctrl_col2 = st.columns([1, 2.5])
        with ctrl_col1:
            st.markdown(
                """
                <div style="display: flex; align-items: center; gap: 8px; margin-top: 6px;">
                    <span style="font-size: 1.25rem;">🖥️</span>
                    <span style="font-size: 1.05rem; font-weight: 700; color: #f8fafc;">Canvas Viewport</span>
                    <span class="badge badge-unit">Responsive Full-Screen</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        with ctrl_col2:
            view_mode = st.radio(
                "Viewport Mode",
                options=["Side-by-Side (Dual)", "Full Canvas (Processed)", "Interactive Split Wipe", "Difference Heatmap"],
                horizontal=True,
                label_visibility="collapsed",
                key="viewport_mode_radio"
            )
        if view_mode == "Side-by-Side (Dual)":
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown(
                    f"<div style='font-size: 0.88rem; font-weight: 700; color: #94a3b8; margin-bottom: 6px;'>📷 SOURCE FRAME ({raw_img.shape[1]} × {raw_img.shape[0]} px)</div>",
                    unsafe_allow_html=True
                )
                st.image(raw_img, width="stretch")

            with col_right:
                st.markdown(
                    f"<div style='font-size: 0.88rem; font-weight: 700; color: #818cf8; margin-bottom: 6px;'>✨ PROCESSED OUTPUT ({proc_img.shape[1]} × {proc_img.shape[0]} px)</div>",
                    unsafe_allow_html=True
                )
                st.image(proc_img, width="stretch")

        elif view_mode == "Interactive Split Wipe":
            split_pos = st.slider(
                "Wipe Position (% Source vs % Output)",
                min_value=0,
                max_value=100,
                value=50,
                step=1,
                key="split_wipe_slider"
            )
            h, w = raw_img.shape[:2]
            proc_aligned = cv2.resize(proc_img, (w, h))
            if raw_img.ndim == 3 and proc_aligned.ndim == 2:
                proc_aligned = cv2.cvtColor(proc_aligned, cv2.COLOR_GRAY2RGB)
            elif raw_img.ndim == 2 and proc_aligned.ndim == 3:
                proc_aligned = cv2.cvtColor(proc_aligned, cv2.COLOR_RGB2GRAY)

            split_col = int((split_pos / 100.0) * w)
            split_canvas = proc_aligned.copy()
            split_canvas[:, :split_col] = raw_img[:, :split_col]

            # Draw white divider boundary line
            if 0 < split_col < w:
                line_w = max(1, w // 256)
                split_canvas[:, max(0, split_col - line_w):min(w, split_col + line_w + 1)] = [255, 255, 255] if split_canvas.ndim == 3 else 255

            st.markdown(
                f"<div style='font-size: 0.88rem; font-weight: 700; color: #818cf8; margin-bottom: 6px;'>⬅️ SOURCE ({split_pos}%) | PROCESSED ({100 - split_pos}%) ➡️</div>",
                unsafe_allow_html=True
            )
            st.image(split_canvas, width="stretch")

        elif view_mode == "Difference Heatmap":
            h, w = raw_img.shape[:2]
            proc_aligned = cv2.resize(proc_img, (w, h))
            if raw_img.ndim == 3 and proc_aligned.ndim == 2:
                proc_aligned = cv2.cvtColor(proc_aligned, cv2.COLOR_GRAY2RGB)
            elif raw_img.ndim == 2 and proc_aligned.ndim == 3:
                proc_aligned = cv2.cvtColor(proc_aligned, cv2.COLOR_RGB2GRAY)

            diff = cv2.absdiff(raw_img, proc_aligned)
            if diff.ndim == 3:
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
            else:
                diff_gray = diff

            heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("<div style='font-size: 0.88rem; font-weight: 700; color: #818cf8; margin-bottom: 6px;'>PROCESSED OUTPUT</div>", unsafe_allow_html=True)
                st.image(proc_img, width="stretch")
            with col_right:
                st.markdown("<div style='font-size: 0.88rem; font-weight: 700; color: #f43f5e; margin-bottom: 6px;'>RESIDUAL ERROR HEATMAP (|Source - Output|)</div>", unsafe_allow_html=True)
                st.image(heatmap_rgb, width="stretch")

        else:
            # Full Canvas (Processed)
            st.markdown(
                f"<div style='font-size: 0.92rem; font-weight: 700; color: #818cf8; margin-bottom: 6px;'>✨ FULL CANVAS PROCESSED OUTPUT ({proc_img.shape[1]} × {proc_img.shape[0]} px)</div>",
                unsafe_allow_html=True
            )
            st.image(proc_img, width="stretch")

def render_histogram_dock(raw_img: np.ndarray, proc_img: np.ndarray):
    """
    Renders synchronized comparative histogram distributions (Input vs Output)
    inside a clean collapsible expander.
    """
    with st.expander("📊 **Comparative Image Histograms (Input vs Output)**", expanded=False):
        hist_mode = st.radio(
            "Channel Selection",
            ["Luminance / Intensity", "RGB Multi-Channel"],
            horizontal=True,
            key="viewport_hist_channel_radio"
        )

        hist_in = compute_histograms(raw_img)
        hist_out = compute_histograms(proc_img)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.2), facecolor='#0e1117')
        for ax in (ax1, ax2):
            ax.set_facecolor('#161b22')
            ax.tick_params(colors='#9ca3af', labelsize=8)
            ax.grid(color='#374151', linestyle='--', linewidth=0.5, alpha=0.6)
            for spine in ax.spines.values():
                spine.set_color('#374151')

        x = np.arange(256)

        if hist_mode == "RGB Multi-Channel" and hist_in["mode"] == "rgb" and hist_out["mode"] == "rgb":
            ax1.plot(x, hist_in["r"], color='#ef4444', label='Red', alpha=0.85, linewidth=1.4)
            ax1.plot(x, hist_in["g"], color='#22c55e', label='Green', alpha=0.85, linewidth=1.4)
            ax1.plot(x, hist_in["b"], color='#3b82f6', label='Blue', alpha=0.85, linewidth=1.4)

            ax2.plot(x, hist_out["r"], color='#ef4444', label='Red', alpha=0.85, linewidth=1.4)
            ax2.plot(x, hist_out["g"], color='#22c55e', label='Green', alpha=0.85, linewidth=1.4)
            ax2.plot(x, hist_out["b"], color='#3b82f6', label='Blue', alpha=0.85, linewidth=1.4)
            ax1.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0', fontsize=8)
            ax2.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0', fontsize=8)
        else:
            ax1.fill_between(x, hist_in["luminance"], color='#818cf8', alpha=0.45)
            ax1.plot(x, hist_in["luminance"], color='#a5b4fc', linewidth=1.4)

            ax2.fill_between(x, hist_out["luminance"], color='#34d399', alpha=0.45)
            ax2.plot(x, hist_out["luminance"], color='#6ee7b7', linewidth=1.4)

        ax1.set_title("Input Frame Distribution", color='#e2e8f0', fontsize=9, pad=6, fontweight='bold')
        ax2.set_title("Processed Output Distribution", color='#e2e8f0', fontsize=9, pad=6, fontweight='bold')
        ax1.set_xlim([0, 255])
        ax2.set_xlim([0, 255])

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
