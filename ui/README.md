# 🖥️ Pixel-Forge — User Interface & Studio Architecture

This directory contains the presentation tier and UI components for **Pixel-Forge**. Built on Streamlit with a custom Digital Content Creation (DCC) dark studio theme, the interface provides a **full-screen responsive dual-viewport canvas**, a **unified suite of 4 workspace expanders** for filter selection, fine-tuning parameters, histograms, and export operations, alongside an active layer stack in the sidebar.

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
│  - Active Stack │  │ 🖥️ RESPONSIVE FULL-SCREEN VIEWPORT CANVAS      │  │
│    (Reorder,    │  │    [Side-by-Side | Full Canvas | Wipe | Heatmap] │  │
│     Bypass,     │  │    - Source Frame vs Processed Output          │  │
│     Delete)     │  │    - Real-Time Radiometric Telemetry Banner    │  │
│  - Clear Stack  │  │    - Zero-overlap native responsive scaling    │  │
│                 │  └────────────────────────────────────────────────┘  │
│                 │                                                      │
│                 │  ┌────────────────────────────────────────────────┐  │
│                 │  │ 🎛️ EXPANDER 1: FILTER CATALOG & ALGORITHM PICK │  │
│                 │  │    - Select Domain & Select Algorithm          │  │
│                 │  │    - Live Micro-Info + Contracts + ➕ Add      │  │
│                 │  └────────────────────────────────────────────────┘  │
│                 │                                                      │
│                 │  ┌────────────────────────────────────────────────┐  │
│                 │  │ ⚙️ EXPANDER 2: ACTIVE FILTER PARAMETER STUDIO   │  │
│                 │  │    - Stage Switcher & Micro-Info Card          │  │
│                 │  │    - Synchronized Sliders & Steppers           │  │
│                 │  │    - Hot-Swap Algorithm & Auto-Convert Shims   │  │
│                 │  └────────────────────────────────────────────────┘  │
│                 │                                                      │
│                 │  ┌────────────────────────────────────────────────┐  │
│                 │  │ 📊 EXPANDER 3: COMPARATIVE HISTOGRAMS          │  │
│                 │  │    - Synchronized Input vs Output Scopes       │  │
│                 │  └────────────────────────────────────────────────┘  │
│                 │                                                      │
│                 │  ┌────────────────────────────────────────────────┐  │
│                 │  │ 💾 EXPANDER 4: EXPORT STUDIO & RECIPES         │  │
│                 │  │    - PNG Image Download & JSON Recipe IO       │  │
│                 │  └────────────────────────────────────────────────┘  │
└─────────────────┴──────────────────────────────────────────────────────┘
```

---

## 🌟 Key UX Design Principles

### 1. 🖥️ Full-Screen Responsive Viewport Canvas
- **Dynamic Proportions:** Images scale dynamically up to `75vh` using CSS flex rules (`object-fit: contain; max-width: 100%;`).
- **Aspect Ratio Integrity:** Eliminates fixed pixel height cropping, allowing portrait, landscape, and panoramic documents to render without distortion or overlapping adjacent widgets.
- **Comparison Modes:**
  1. **Side-by-Side (Dual):** Simultaneous high-res comparison between source and output frames.
  2. **Full Canvas (Processed):** High-definition full-canvas view of processed output.
  3. **Interactive Split Wipe:** Dynamic horizontal reveal slider between input and output with a 2px white dividing line.
  4. **Difference Heatmap:** Color-mapped thermal error magnitude ($|I_{\text{source}} - I_{\text{processed}}|$) highlighting exact pixel modifications.

### 2. 🎛️ Unified Expander Suite
All controls below the viewport canvas are organized into four uniform, collapsible `st.expander` containers:
- **🎛️ Filter Catalog & Algorithm Selector:**
  - Select Domain from standard syllabus & DCC categories.
  - Select Algorithm from the domain's tools.
  - Live preview card displaying: `💡 What it does`, `🎯 Practical Example`, contract specifications (`In` $\to$ `Out`).
  - 1-Click `➕ Add to Pipeline Stack` button.
- **⚙️ Active Filter Parameters & Stage Controls:**
  - Stage switcher dropdown allowing instant switching between active pipeline layers.
  - Auto-convert shims toggle and per-stage parameter reset.
  - Algorithm hot-swapping within the active domain.
  - Synchronized dual slider and numeric input boxes for micro-precision adjustment.
- **📊 Comparative Image Histograms:**
  - Synchronized input vs. output distributions with Luminance and RGB multi-channel toggles.
- **💾 Export Studio & Pipeline Recipe:**
  - 1-click lossless full-resolution PNG export.
  - JSON pipeline recipe serialization and upload for reproducible experiments.

### 3. 🎞️ Streamlined Sidebar Pipeline Stack
- **Dedicated Layer Management:** Sidebar focuses exclusively on the active layer sequence:
  - Preset loader dropdown with loop-prevention callbacks.
  - Individual stage cards with live bypass toggles, priority reordering (`▲`/`▼`), parameter selection (`⚙️`), and removal (`✕`).
  - Clear All button to reset stack cleanly.
