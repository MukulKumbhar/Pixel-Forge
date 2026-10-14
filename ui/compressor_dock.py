"""
Interactive image compressor UI component supporting lossless and deep lossy modes.
"""

import time
from typing import Optional
import cv2
import numpy as np
import streamlit as st

from algorithms.compression import compress_image_stream


def _format_bytes(num_bytes: int) -> str:
    """Format byte counts into human-readable KB or MB."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024.0:.1f} KB"
    else:
        return f"{num_bytes / (1024.0 * 1024.0):.2f} MB"


def render_compressor_dock(raw_image: np.ndarray, processed_image: np.ndarray, master_image: Optional[np.ndarray] = None):
    """Render image compressor dock with lossless and lossy slider controls."""
    with st.expander("🗜️ **Pro Image Compressor (Lossless & Deep Lossy with Quality Slider)**", expanded=False):
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div>
                    <h3 style="margin: 0; font-size: 1.15rem; color: #f8fafc; font-weight: 700;">Global-Grade Image Compression Engine</h3>
                    <p style="margin: 2px 0 0 0; font-size: 0.82rem; color: #94a3b8;">
                        Two-way architecture: Bit-exact <b>Lossless</b> encoding or <b>Deep Lossy</b> compression with real-time slider controls.
                    </p>
                </div>
                <div>
                    <span class="badge badge-unit">Lossless & Deep Lossy</span>
                    <span class="badge badge-rgb" style="margin-left: 4px;">WebP / JPEG / PNG</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        c_src, c_mode = st.columns([1.1, 1.9])
        with c_src:
            src_options = ["Current Processed Output", "Original Raw Frame"]
            if master_image is not None and master_image.shape[:2] != raw_image.shape[:2]:
                src_options.insert(0, f"Full-Resolution Master Frame ({master_image.shape[1]} × {master_image.shape[0]} px)")

            source_choice = st.radio(
                "Image Source",
                options=src_options,
                horizontal=True,
                key="compressor_src_radio"
            )
            if "Full-Resolution Master" in source_choice:
                target_image = master_image
            elif source_choice == "Current Processed Output":
                target_image = processed_image
            else:
                target_image = raw_image

        with c_mode:
            compression_mode = st.radio(
                "Compression Architecture",
                options=[
                    "💎 Lossless (Zero Quality Loss)",
                    "🎚️ Deep Lossy (Controlled via Quality Slider)"
                ],
                horizontal=True,
                key="compressor_arch_radio"
            )

        is_lossless = "Lossless" in compression_mode
        st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255, 255, 255, 0.08); margin: 12px 0;'>", unsafe_allow_html=True)

        if is_lossless:
            l_col1, l_col2, l_col3 = st.columns([1.2, 1.4, 1.4])
            with l_col1:
                lossless_format = st.selectbox(
                    "Lossless Codec",
                    options=["WebP (VP8L Lossless - Recommended)", "PNG (Optimized Deflate)"],
                    key="comp_lossless_codec_select"
                )
                fmt_key = "webp" if "WebP" in lossless_format else "png"

            with l_col2:
                max_effort = 6 if fmt_key == "webp" else 9
                effort = st.slider(
                    "Compression Effort (CPU Optimization)",
                    min_value=1,
                    max_value=max_effort,
                    value=max_effort,
                    step=1,
                    key="comp_lossless_effort_slider",
                    help="Higher effort analyzes spatial patterns more exhaustively to reduce file size without any quality loss."
                )

            with l_col3:
                strip_meta = st.checkbox(
                    "Strip EXIF / Metadata Chunks",
                    value=True,
                    key="comp_lossless_strip_meta",
                    help="Removes camera tags, GPS, and color profile bloat to shave extra kilobytes."
                )

            st.markdown(
                """
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 10px 14px; margin: 8px 0 14px 0;">
                    <span style="color: #34d399; font-weight: 700; font-size: 0.88rem;">🛡️ Mathematical Lossless Guarantee:</span>
                    <span style="color: #cbd5e1; font-size: 0.84rem; margin-left: 6px;">
                        PSNR = <b>99.9 dB</b> | SSIM = <b>1.000</b> | MSE = <b>0.000</b> | Exactly 100% of the pixel information is preserved.
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

            comp_result = compress_image_stream(
                image=target_image,
                mode="lossless",
                format_choice=fmt_key,
                effort=effort,
                scale_factor=1.0,
                strip_metadata=strip_meta
            )

        else:
            y_col1, y_col2, y_col3, y_col4 = st.columns([1.2, 1.8, 1.1, 1.1])
            with y_col1:
                lossy_format = st.selectbox(
                    "Lossy Codec",
                    options=[
                        "WebP (Next-Gen Modern - High Ratio)",
                        "JPEG (Standard DCT)",
                        "Paletted PNG (Color Quantized)"
                    ],
                    key="comp_lossy_codec_select"
                )
                if "WebP" in lossy_format:
                    fmt_key = "webp"
                elif "JPEG" in lossy_format:
                    fmt_key = "jpeg"
                else:
                    fmt_key = "png_quantized"

            with y_col2:
                quality_val = st.slider(
                    "Compression Quality Level (Slider)",
                    min_value=1,
                    max_value=100,
                    value=75,
                    step=1,
                    key="comp_lossy_quality_slider",
                    help="Lower quality applies deeper quantization and higher compression, drastically reducing file size."
                )

                if quality_val >= 88:
                    tier_badge = "<span style='color: #34d399; font-weight: 700;'>Studio Master (Near-Lossless, 2x — 5x smaller)</span>"
                elif quality_val >= 65:
                    tier_badge = "<span style='color: #38bdf8; font-weight: 700;'>Web & Mobile (Recommended for Global Web, 6x — 15x smaller)</span>"
                elif quality_val >= 35:
                    tier_badge = "<span style='color: #fbbf24; font-weight: 700;'>Aggressive Bandwidth Saver (16x — 35x smaller)</span>"
                else:
                    tier_badge = "<span style='color: #f43f5e; font-weight: 700;'>Deep Compression Mode (Ultra-low KB for Fast Delivery, 35x — 100x+ smaller)</span>"

                st.markdown(f"<div style='font-size: 0.78rem; margin-top: -8px;'>Preset Tier: {tier_badge}</div>", unsafe_allow_html=True)

            with y_col3:
                scale_options = {
                    1.0: "100% (Original)",
                    0.75: "75% (Downscale)",
                    0.50: "50% (Half-Res)",
                    0.35: "35% (Deep Min)"
                }
                scale_factor = st.selectbox(
                    "Dimension Scale",
                    options=list(scale_options.keys()),
                    format_func=lambda k: scale_options[k],
                    key="comp_lossy_scale_select",
                    help="Scale canvas dimensions to achieve extreme deep compression down to tiny KB files."
                )

            with y_col4:
                if fmt_key == "jpeg":
                    chroma = st.selectbox("Chroma Sampling", ["4:2:0", "4:4:4"], key="comp_lossy_chroma_select")
                    num_colors = 64
                elif fmt_key == "png_quantized":
                    num_colors = st.slider("Palette Colors", 2, 256, 32, step=2, key="comp_palette_colors_slider")
                    chroma = "4:2:0"
                else:
                    chroma = "4:2:0"
                    num_colors = 64
                    effort_lossy = st.slider("Effort / Method", 1, 6, 6, key="comp_webp_lossy_effort")

            comp_result = compress_image_stream(
                image=target_image,
                mode="lossy",
                format_choice=fmt_key,
                quality=quality_val,
                effort=6 if fmt_key != "webp" else st.session_state.get("comp_webp_lossy_effort", 6),
                scale_factor=scale_factor,
                chroma_subsampling=chroma,
                num_palette_colors=num_colors
            )

        orig_sz_str = _format_bytes(comp_result["orig_size_bytes"])
        comp_sz_str = _format_bytes(comp_result["compressed_size_bytes"])
        saved_pct = comp_result["space_saved_percent"]
        ratio = comp_result["compression_ratio"]

        st.markdown(
            f"""
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin: 12px 0;">
                <div class="telemetry-card">
                    <div class="telemetry-label">📦 Original Size</div>
                    <div class="telemetry-value" style="color: #94a3b8;">{orig_sz_str}</div>
                </div>
                <div class="telemetry-card">
                    <div class="telemetry-label">🗜️ Compressed Size</div>
                    <div class="telemetry-value" style="color: #38bdf8; font-weight: 800;">{comp_sz_str}</div>
                </div>
                <div class="telemetry-card">
                    <div class="telemetry-label">⚡ Space Saved</div>
                    <div class="telemetry-value" style="color: #34d399; font-weight: 800;">{saved_pct:.1f}%</div>
                </div>
                <div class="telemetry-card">
                    <div class="telemetry-label">📊 Ratio</div>
                    <div class="telemetry-value" style="color: #a78bfa;">{ratio:.1f} : 1</div>
                </div>
                <div class="telemetry-card">
                    <div class="telemetry-label">📈 PSNR</div>
                    <div class="telemetry-value" style="color: #fbbf24;">{comp_result['psnr']:.1f} dB</div>
                </div>
                <div class="telemetry-card">
                    <div class="telemetry-label">🔍 SSIM</div>
                    <div class="telemetry-value" style="color: #f472b6;">{comp_result['ssim']:.3f}</div>
                </div>
                <div class="telemetry-card">
                    <div class="telemetry-label">⏱️ Encoding Time</div>
                    <div class="telemetry-value">{comp_result['elapsed_ms']:.1f} ms</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        preview_col1, preview_col2 = st.columns(2)
        with preview_col1:
            st.markdown(
                f"<div style='font-size: 0.85rem; font-weight: 700; color: #94a3b8; margin-bottom: 4px;'>📷 UNCOMPRESSED INPUT ({target_image.shape[1]} × {target_image.shape[0]} px • {orig_sz_str})</div>",
                unsafe_allow_html=True
            )
            st.image(target_image, width="stretch")

        with preview_col2:
            reconstructed = comp_result["decompressed_image"]
            st.markdown(
                f"<div style='font-size: 0.85rem; font-weight: 700; color: #38bdf8; margin-bottom: 4px;'>✨ COMPRESSED OUTPUT ({reconstructed.shape[1]} × {reconstructed.shape[0]} px • {comp_sz_str})</div>",
                unsafe_allow_html=True
            )
            st.image(reconstructed, width="stretch")

        st.write("")
        dl_col1, dl_col2, dl_col3 = st.columns([1, 2, 1])
        with dl_col2:
            ext = comp_result["file_extension"]
            filename = f"pixel_forge_compressed.{ext}"
            st.download_button(
                label=f"⬇️ Download Compressed Image ({ext.upper()} • {comp_sz_str} • {saved_pct:.1f}% Saved)",
                data=comp_result["compressed_bytes"],
                file_name=filename,
                mime=comp_result["mime_type"],
                use_container_width=True,
                type="primary"
            )
