# 🖥️ ImageLab Pro — User Interface & Studio Architecture

This directory contains the presentation tier and UI components for **ImageLab Pro**. Built on Streamlit with a custom Digital Content Creation (DCC) dark studio theme, the interface provides a **permanently fixed visual preview** and a **unified resizable, collapsible bottom tray** for all filter controls.

---

## 🏛️ Studio Layout Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        STUDIO HEADER & BENCHMARK LOADER                │
├─────────────────┬──────────────────────────────────────────────────────┤
│  SIDEBAR:       │  MAIN WORKSPACE:                                     │
│  - Presets      │                                                      │
│    (Dropdown &  │  ┌────────────────────────────────────────────────┐  │
│     1-Click)    │  │ 🖥️ PINNED TOP DUAL-VIEWPORT PREVIEW CANVAS     │  │
│  - Active Stack │  │    - Source Frame vs Processed Output / Wipe   │  │
│    (Reorder,    │  │    - Real-Time Radiometric Telemetry Banner    │  │
│     Bypass,     │  │    - Always fixed in sight; never scrolls off! │  │
│     Delete)     │  └────────────────────────────────────────────────┘  │
│  - 1-Click      │                                                      │
│    Quick        │  ┌────────────────────────────────────────────────┐  │
│    Presets      │  │ 🎛️ UNIFIED RESIZABLE & COLLAPSIBLE BOTTOM TRAY │  │
│                 │  │    [ 🔽 Hide Tray ]  [ 📏 Compact / Std / Exp ]│  │
│                 │  │  ┌──────────────────────────────────────────┐  │  │
│                 │  │  │  INTERNAL SCROLL ONLY:                   │  │  │
│                 │  │  │  [Tab 1: ⚙️ Active Stage Sliders]        │  │  │
│                 │  │  │  [Tab 2: 🗂️ Browse Catalog (66 Tools)]   │  │  │
│                 │  │  │  [Tab 3: 📊 Comparative Histograms]      │  │  │
│                 │  │  │  [Tab 4: 💾 Export PNG & Recipe JSON]    │  │  │
│                 │  │  └──────────────────────────────────────────┘  │  │
│                 │  │  (Native drag handle: stretch / shrink height)  │  │
│                 │  └────────────────────────────────────────────────┘  │
└─────────────────┴──────────────────────────────────────────────────────┘
```

---

## 🌟 Key UX Innovations

### 1. 📌 Permanently Fixed Image Preview
- **The Design:** The preview panel sits at the top of the workspace. Preview images are bounded via CSS (`max-height: 380px !important; object-fit: contain !important;`), ensuring that the entire preview canvas always fits comfortably on screen without ever pushing the page down.
- **Viewing Modes:**
  1. **Side-by-Side:** Direct simultaneous comparison between source and output frames.
  2. **Interactive Split Wipe:** Real-time slider revealing source vs. output with a crisp divider line.
  3. **Difference Heatmap:** Color-mapped residual error ($|I_{\text{source}} - I_{\text{processed}}|$) highlighting structural modifications.
  4. **Output Only:** Clean view of the final processed raster.

### 2. 🎛️ Unified Resizable & Collapsible Bottom Tray
- **All Filter Data in One Place:** Aside from the preview panel, **all data about filters resides in the bottom tray**:
  - **Tab 1 (⚙️ Active Filter Settings):** Sliders, numeric steppers, algorithm hot-swap, and compact micro-info cards (`💡 What it does`, `🎯 Example`).
  - **Tab 2 (🗂️ Browse Filter Catalog):** 66 real-time tools categorized across 7 domain tabs with 1-click `➕ Add Block` buttons.
  - **Tab 3 (📊 Comparative Histograms):** Dual input vs. output histograms with Luminance and RGB multi-channel toggles.
  - **Tab 4 (💾 Export Studio & Recipe):** 1-click PNG image export and JSON pipeline recipe import/export.
- **Internal Independent Scroll:** The bottom tray has its own internal scrolling (`overflow-y: auto;`). Scrolling through filters or parameters scrolls **only the tray**—the image preview above never scrolls out of view!

### 3. 👁️ 1-Click Hide / Collapse (Full-Screen Preview)
- Clicking `🔽 Hide Tray` instantly minimizes the bottom tray into a slim 36px floating bar.
- This gives the image preview 100% of the screen height, allowing you to inspect the final processed image without any obstructions.
- Clicking `🔼 Show Filter Tray` immediately restores the controls.

### 4. 📏 Manually Scalable (Stretch & Shrink Height)
- **Native Drag Resize:** The tray container is equipped with CSS `resize: vertical;`, enabling users to click and drag the bottom corner to stretch or shrink its height dynamically.
- **1-Click Height Presets:** Quick preset buttons in the tray header bar allow instantaneous height switching:
  - **Compact (250px):** Minimal footprint for small laptop displays.
  - **Standard (380px):** Balanced height for regular parameter tweaking.
  - **Expanded (520px):** Large canvas for browsing the 66-tool catalog.

### 5. ⚡ Zero-Loop Preset Execution & Anti-Flicker Architecture
- Presets use an explicit `on_change` callback that resets the widget selection back to default index (`"-- Select Preset --"`) immediately upon execution, completely eliminating recursive rerun loops.
- 1-Click Quick Preset buttons (`✨ Retouch`, `🎨 Sketch`, `🎬 Film Grade`, `📄 Doc Clean`, `🔬 Denoise`) allow instant one-tap workflow application.
- Injected CSS rules (`transition: none !important; animation: none !important; opacity: 1 !important;`) suppress Streamlit's default page dimming during reruns for instantaneous feedback.
