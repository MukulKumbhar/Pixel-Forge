"""
ImageLab Pro - Studio Styling & Design System
Applies high-end dark-mode aesthetic with glassmorphism, responsive high-resolution
dual-viewport canvas, spacious independently scrollable bottom tray, and modern typography.
"""

from typing import Optional
import streamlit as st


BASE_STUDIO_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Base Dark DCC Studio Theme */
:root {
    --bg-primary: #0e1117;
    --bg-secondary: #161b22;
    --bg-panel: rgba(22, 27, 34, 0.85);
    --border-subtle: rgba(255, 255, 255, 0.1);
    --border-accent: rgba(99, 102, 241, 0.5);
    --accent-indigo: #6366f1;
    --accent-cyan: #06b6d4;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --text-primary: #f8fafc;
    --text-muted: #94a3b8;
}

html, body, [class*="css"], [data-testid="stMarkdownContainer"] {
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    letter-spacing: -0.01em;
}

/* Overall page padding - spacious and breathable */
.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 96%;
}

/* Suppress jarring Streamlit rerun opacity transitions & page fading */
[data-testid="stAppViewContainer"] .main,
[data-testid="stMain"],
div[data-testid="stVerticalBlock"] > div,
.element-container {
    transition: none !important;
    animation: none !important;
}

.stApp [data-testid="stMain"] {
    opacity: 1 !important;
}

[data-testid="stAppViewContainer"] {
    opacity: 1 !important;
}

/* Header Banner - Spacious Studio Bar */
.studio-header {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 0.9rem 1.6rem;
    margin-bottom: 1.2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 6px 25px 0 rgba(0, 0, 0, 0.45);
}

.studio-title-box {
    display: flex;
    align-items: center;
    gap: 12px;
}

.studio-title {
    font-size: 1.75rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.025em;
    margin: 0;
    line-height: 1.2;
}

.studio-subtitle {
    font-size: 0.88rem;
    color: var(--text-muted);
    margin: 3px 0 0 0;
}

/* Viewport Image Fit Container - Dynamic Preview Height */
[data-testid="stImage"] img {
    max-height: __PREVIEW_HEIGHT__px !important;
    width: auto !important;
    margin: 0 auto !important;
    object-fit: contain !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5) !important;
}

.viewport-frame {
    display: flex;
    justify-content: center;
    align-items: center;
    background: #090d14;
    border-radius: 12px;
    border: 1px solid var(--border-subtle);
    overflow: hidden;
    max-height: __PREVIEW_HEIGHT__px !important;
}

.viewport-frame img {
    max-height: __PREVIEW_HEIGHT__px !important;
    width: auto;
    object-fit: contain;
}

/* Telemetry Diagnostics Grid */
.telemetry-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 12px;
    margin-bottom: 1.1rem;
}

.telemetry-card {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    text-align: left;
    transition: all 0.2s ease-in-out;
}

.telemetry-card:hover {
    border-color: var(--border-accent);
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
}

.telemetry-label {
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    margin-bottom: 4px;
    font-weight: 600;
}

.telemetry-value {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
}

/* Layer Stack Cards */
.layer-card {
    background: rgba(30, 41, 59, 0.55);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.7rem;
    transition: all 0.15s ease;
}

.layer-card.active-layer {
    border-color: var(--accent-indigo);
    background: rgba(49, 46, 129, 0.3);
    box-shadow: 0 0 16px rgba(99, 102, 241, 0.25);
}

.layer-card.bypassed {
    opacity: 0.45;
    border-style: dashed;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 0.22rem 0.6rem;
    border-radius: 6px;
    font-size: 0.74rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    white-space: nowrap;
}

.badge-rgb {
    background: rgba(16, 185, 129, 0.18);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
}

.badge-gray {
    background: rgba(148, 163, 184, 0.18);
    color: #cbd5e1;
    border: 1px solid rgba(148, 163, 184, 0.35);
}

.badge-binary {
    background: rgba(245, 158, 11, 0.18);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.35);
}

.badge-unit {
    background: rgba(99, 102, 241, 0.18);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.35);
}

.badge-conflict {
    background: rgba(244, 63, 94, 0.2);
    color: #fda4af;
    border: 1px solid rgba(244, 63, 94, 0.4);
}

/* Filter Micro-Info Card */
.filter-micro-info {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-left: 4px solid #6366f1;
    border-radius: 10px;
    padding: 0.95rem 1.4rem;
    font-size: 0.95rem;
    color: #e2e8f0;
    margin-bottom: 1.2rem;
    line-height: 1.6;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.3);
}

.filter-micro-info .info-title {
    color: #93c5fd;
    font-weight: 700;
}

.filter-micro-info .example-title {
    color: #34d399;
    font-weight: 700;
}

/* Studio Unified Resizable & Collapsible Bottom Tray */
.studio-bottom-tray {
    background: #0b1120;
    border: 1px solid rgba(99, 102, 241, 0.45);
    border-radius: 14px;
    box-shadow: 0 -10px 40px rgba(0, 0, 0, 0.65);
    margin-top: 1.1rem;
    margin-bottom: 2rem;
    overflow: hidden;
    resize: vertical;
    min-height: 280px;
    max-height: 85vh;
}

.tray-scroll-body {
    overflow-y: auto;
    overflow-x: hidden;
    padding: 1.2rem 1.6rem;
    scrollbar-width: thin;
    scrollbar-color: #6366f1 rgba(15, 23, 42, 0.8);
}

.tray-scroll-body::-webkit-scrollbar {
    width: 7px;
}

.tray-scroll-body::-webkit-scrollbar-thumb {
    background-color: #6366f1;
    border-radius: 4px;
}

.tray-minimized-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(30, 41, 59, 0.85);
    border: 1px dashed rgba(99, 102, 241, 0.45);
    border-radius: 10px;
    padding: 0.7rem 1.4rem;
    margin-top: 0.8rem;
    margin-bottom: 1.5rem;
}

/* Category Dock Card - Spacious & Readable */
.dock-card {
    background: rgba(30, 41, 59, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 1rem 1.15rem;
    min-height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    margin-bottom: 0.6rem;
}

.dock-card:hover {
    border-color: #818cf8;
    background: rgba(49, 46, 129, 0.35);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.25);
}

.dock-card-title {
    font-weight: 700;
    font-size: 1.02rem;
    color: #f8fafc;
    letter-spacing: -0.01em;
    margin-bottom: 5px;
}

.dock-card-desc {
    font-size: 0.84rem;
    color: #cbd5e1;
    line-height: 1.5;
    margin-bottom: 10px;
}

/* Preset Quick Action Pills */
.preset-quick-btn {
    display: inline-block;
    padding: 0.4rem 0.8rem;
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 8px;
    color: #e2e8f0;
    font-size: 0.84rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
    text-align: center;
    margin-bottom: 0.5rem;
}

.preset-quick-btn:hover {
    background: rgba(99, 102, 241, 0.35);
    border-color: #818cf8;
    color: #ffffff;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.88rem;
    transition: all 0.15s ease;
}

/* Streamlit container clean borders */
div[data-testid="stExpander"] {
    background: rgba(30, 41, 59, 0.3);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
}
</style>
"""


def apply_studio_styles(preview_height: Optional[int] = None):
    """
    Injects custom CSS theme into Streamlit runtime with dynamic preview height and spacious typography.
    """
    if preview_height is None:
        preview_height = st.session_state.get("preview_height", 520)

    css_content = BASE_STUDIO_CSS.replace("__PREVIEW_HEIGHT__", str(preview_height))
    st.markdown(css_content, unsafe_allow_html=True)
