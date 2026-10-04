"""
Pixel-Forge - Studio Styling & Design System
Applies high-end dark-mode aesthetic with glassmorphism, responsive full-width
dual-viewport canvas, clean typography, and zero-overlap layout.
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

/* Overall page padding - spacious, breathable, zero overlap */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 3rem;
    max-width: 98%;
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
    padding: 0.85rem 1.4rem;
    margin-bottom: 1rem;
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
    font-size: 1.65rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.025em;
    margin: 0;
    line-height: 1.2;
}

.studio-subtitle {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin: 3px 0 0 0;
}

/* Viewport Image Fit Container - Responsive Full Screen Viewport */
[data-testid="stImage"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
}

[data-testid="stImage"] img {
    max-width: 100% !important;
    height: auto !important;
    max-height: 75vh !important;
    object-fit: contain !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45) !important;
}

/* Telemetry Diagnostics Grid */
.telemetry-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(125px, 1fr));
    gap: 10px;
    margin-bottom: 1rem;
}

.telemetry-card {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 0.65rem 0.9rem;
    text-align: left;
    transition: all 0.2s ease-in-out;
}

.telemetry-card:hover {
    border-color: var(--border-accent);
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
}

.telemetry-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    margin-bottom: 3px;
    font-weight: 600;
}

.telemetry-value {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
}

/* Layer Stack Cards in Sidebar */
.layer-card {
    background: rgba(30, 41, 59, 0.55);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 0.75rem 0.9rem;
    margin-bottom: 0.5rem;
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
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
    font-size: 0.72rem;
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
    padding: 0.85rem 1.25rem;
    font-size: 0.92rem;
    color: #e2e8f0;
    margin-bottom: 1rem;
    line-height: 1.55;
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

/* Parameter Drawer Panel Container */
.param-drawer, .param-studio-panel {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    margin-top: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.86rem;
    transition: all 0.15s ease;
}

/* Studio Expanders Suite - Polished Dark DCC Aesthetics */
div[data-testid="stExpander"] {
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 12px !important;
    margin-top: 0.6rem !important;
    margin-bottom: 0.9rem !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
    overflow: hidden !important;
    transition: all 0.2s ease-in-out !important;
}

div[data-testid="stExpander"]:hover {
    border-color: rgba(99, 102, 241, 0.55) !important;
    box-shadow: 0 6px 24px rgba(99, 102, 241, 0.2) !important;
}

div[data-testid="stExpander"] summary {
    font-weight: 700 !important;
    font-size: 0.96rem !important;
    color: #f8fafc !important;
    padding: 0.75rem 1rem !important;
    letter-spacing: -0.01em !important;
}

div[data-testid="stExpander"] summary:hover {
    color: #a5b4fc !important;
}

div[data-testid="stExpander"] div[role="region"] {
    padding: 0.8rem 1.2rem 1.2rem 1.2rem !important;
}
</style>
"""


def apply_studio_styles(preview_height: Optional[int] = None):
    """
    Injects custom CSS theme into Streamlit runtime for clean, responsive, non-overlapping studio UI.
    """
    st.markdown(BASE_STUDIO_CSS, unsafe_allow_html=True)
