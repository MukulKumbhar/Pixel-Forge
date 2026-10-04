# 🖥️ Pixel-Forge — User Interface & Studio Architecture

This directory contains the presentation tier and UI components for **Pixel-Forge**. Built on Streamlit with a custom Digital Content Creation (DCC) dark studio theme, the interface provides a **full-screen responsive dual-viewport canvas**, **in-situ parameter editing with micro-info cards**, and **synchronized radiometric telemetry**.

---

## 🏛️ Studio Layout Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        STUDIO HEADER & BENCHMARK LOADER                │
│    [Sample Selectbox]   [Upload Custom Image]   [↩ Undo ↪ Redo 🗑️ Reset] │
├─────────────────┬──────────────────────────────────────────────────────┤
│  SIDEBAR:       │  MAIN WORKSPACE:                                     │
│  - Presets      │                                                      │
│    Dropdown     │  ┌────────────────────────────────────────────────┐  │
│  - ➕ Add Stage │  │ 🖥️ RESPONSIVE FULL-SCREEN VIEWPORT CANVAS      │  │
│    (Domain &    │  │    [Side-by-Side | Full Canvas | Wipe | Heatmap] │  │
│     Algo Pick)  │  │    - Source Frame vs Processed Output          │  │
│  - Active Stack │  │    - Real-Time Radiometric Telemetry Banner    │  │
│    (Reorder,    │  │    - Zero-overlap native responsive scaling    │  │
│     Bypass,     │  └────────────────────────────────────────────────┘  │
│     Delete)     │                                                      │
│  - Clear Stack  │  ┌────────────────────────────────────────────────┐  │
│                 │  │ ⚙️ ACTIVE STAGE PARAMETER STUDIO               │  │
│                 │  │    - 💡 Micro-Info Card (What it does + Example)│  │
│                 │  │    - Synchronized Sliders & Steppers           │  │
│                 │  │    - Hot-Swap Algorithm & Auto-Convert Shims   │  │
│                 │  └────────────────────────────────────────────────┘  │
│                 │                                                      │
│                 │  ┌────────────────────────────────────────────────┐  │
│                 │  │ 📊 COMPARATIVE HISTOGRAMS (Input vs Output)    │  │
│                 │  └────────────────────────────────────────────────┘  │
│                 │  ┌────────────────────────────────────────────────┐  │
│                 │  │ 💾 EXPORT STUDIO & PIPELINE RECIPE (PNG/JSON)  │  │
│                 │  └────────────────────────────────────────────────┘  │
└─────────────────┴──────────────────────────────────────────────────────┘
```

---

## 🌟 Key UX Design Principles

### 1. 🖥️ Full-Screen Responsive Viewport Canvas
- **Dynamic Proportions:** Images scale dynamically up to `75vh` using CSS flex rules (`object-fit: contain; max-width: 100%;`).
- **Aspect Ratio Integrity:** Eliminates fixed pixel height cropping (`Compact 400px`, `Comfortable 520px`, etc.), allowing natural portrait, landscape, or panoramic documents to render without distortion or overlapping adjacent widgets.
- **Comparison Modes:**
  1. **Side-by-Side (Dual):** Simultaneous high-res comparison between source and output frames.
  2. **Full Canvas (Processed):** High-definition full-canvas view of processed output.
  3. **Interactive Split Wipe:** Dynamic horizontal reveal slider between input and output with a 2px white dividing line.
  4. **Difference Heatmap:** Color-mapped thermal error magnitude ($|I_{\text{source}} - I_{\text{processed}}|$) highlighting exact pixel modifications.

### 2. 🎞️ Streamlined Sidebar Pipeline Stack & Tool Adder
- **No Clutter, Zero Redundancy:** Removed redundant QuickPreset buttons in favor of an instantaneous, state-resetting dropdown preset manager.
- **Direct Stage Addition:** Browse all 7 domains and 66 algorithms directly from the sidebar. Clicking `➕ Add Stage to Stack` immediately appends the stage to the active chain and opens its parameters.
- **Stage Management:** Individual stage controls for live bypass toggling, priority reordering (`▲`/`▼`), parameter editing (`⚙️`), and removal (`✕`).

### 3. ⚙️ Active Stage Parameter Studio & Micro-Info Cards
- Displays directly below the viewport canvas when a layer is selected.
- **💡 What it does:** Concise 1-sentence technical definition.
- **🎯 Example:** Real-world practical use case.
- **Synchronized Controls:** Dual slider and numeric input boxes for micro-precision parameter adjustment.

### 4. 📊 Scopes & Export Studio
- **Comparative Histograms:** Collapsible drawer rendering side-by-side luminance and RGB multi-channel curves.
- **Export Studio:** 1-click lossless PNG raster download and pipeline JSON recipe import/export for reproducible workflows.
