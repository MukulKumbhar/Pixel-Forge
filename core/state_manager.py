"""
ImageLab Pro - Session State & History Manager
Handles layer stack lifecycle (add, remove, reorder, swap, toggle),
parameter updates, undo/redo history stacks, and industry workflow presets.
"""

import copy
import uuid
from typing import Any, Dict, List, Optional
import streamlit as st
from algorithms.registry import ALGORITHM_REGISTRY, DOMAINS

# Industry & Creative Workflow Presets
PRESETS: Dict[str, Dict[str, Any]] = {
    "photoshop_portrait": {
        "title": "✨ Photoshop Studio: Glamour Skin Retouch",
        "description": "Bilateral skin smoothing, smart vibrance boost, subtle unsharp masking, and soft optical vignette.",
        "layers": [
            {
                "algorithm_key": "rest_bilateral",
                "params": {"d": 9, "sigma_color": 75.0, "sigma_space": 75.0},
                "auto_convert": True
            },
            {
                "algorithm_key": "col_vibrance",
                "params": {"vibrance": 0.40, "saturation": 0.10},
                "auto_convert": True
            },
            {
                "algorithm_key": "rest_unsharp_mask",
                "params": {"amount": 1.2, "radius": 3, "threshold": 2},
                "auto_convert": True
            },
            {
                "algorithm_key": "art_vignette",
                "params": {"radius": 0.85, "intensity": 0.45},
                "auto_convert": True
            }
        ]
    },
    "artistic_sketch": {
        "title": "🎨 Fine Art: Graphite Pencil Drawing",
        "description": "Blends inverted Gaussian blur with grayscale raster via Color Dodge and adds dramatic border shading.",
        "layers": [
            {
                "algorithm_key": "art_pencil_sketch",
                "params": {"ksize": 21, "color_mode": "pencil_gray"},
                "auto_convert": True
            },
            {
                "algorithm_key": "art_vignette",
                "params": {"radius": 0.80, "intensity": 0.60},
                "auto_convert": True
            }
        ]
    },
    "pop_art_cartoon": {
        "title": "🖍️ Pop Art: Graphic Cel-Shaded Cartoon",
        "description": "Multi-pass bilateral color flattening, dark ink line contours, and discrete color posterization.",
        "layers": [
            {
                "algorithm_key": "art_cartoonify",
                "params": {"num_bilateral": 3, "d": 9, "sigma_color": 75.0, "edge_block_size": 9},
                "auto_convert": True
            },
            {
                "algorithm_key": "rad_posterize",
                "params": {"levels": 5},
                "auto_convert": True
            }
        ]
    },
    "retro_vintage": {
        "title": "📷 Retro Nostalgia: Antique Sepia & Grain",
        "description": "Warm antique sepia silver print tone, subtle film grain noise, and circular optical vignette.",
        "layers": [
            {
                "algorithm_key": "col_sepia",
                "params": {"strength": 0.90},
                "auto_convert": True
            },
            {
                "algorithm_key": "rest_noise_gaussian",
                "params": {"mean": 0.0, "sigma": 15.0},
                "auto_convert": True
            },
            {
                "algorithm_key": "art_vignette",
                "params": {"radius": 0.70, "intensity": 0.70},
                "auto_convert": True
            }
        ]
    },
    "film_grade": {
        "title": "🎬 Cinema Film Grade & Tone Curve",
        "description": "Piecewise dynamic contrast stretch followed by von Kries warm white balance and 3-way Lift/Gamma/Gain split toning.",
        "layers": [
            {
                "algorithm_key": "rad_contrast_stretch",
                "params": {"r1": 60, "s1": 15, "r2": 190, "s2": 240},
                "auto_convert": True
            },
            {
                "algorithm_key": "col_white_balance",
                "params": {"temperature": 0.15, "tint": -0.05},
                "auto_convert": True
            },
            {
                "algorithm_key": "col_lift_gamma_gain",
                "params": {"lift": 0.05, "gamma": 1.10, "gain": 1.15},
                "auto_convert": True
            },
            {
                "algorithm_key": "col_hsi_vector",
                "params": {"hue_shift": 5.0, "sat_scale": 1.20, "intensity_offset": 0.0},
                "auto_convert": True
            }
        ]
    },
    "doc_cleaning": {
        "title": "📄 Document Binarization & OCR Prep",
        "description": "Gaussian smoothing, high-boost sharpening, Otsu binarization, and morphological noise opening.",
        "layers": [
            {
                "algorithm_key": "rest_arithmetic_mean",
                "params": {"ksize": 3},
                "auto_convert": True
            },
            {
                "algorithm_key": "seg_high_boost",
                "params": {"boost_a": 1.6, "ksize": 5},
                "auto_convert": True
            },
            {
                "algorithm_key": "seg_otsu",
                "params": {},
                "auto_convert": True
            },
            {
                "algorithm_key": "morph_open",
                "params": {"shape": "rect", "ksize": 3},
                "auto_convert": True
            }
        ]
    },
    "edge_hough": {
        "title": "📐 Structural Edge & Hough Line Geometry",
        "description": "Median noise suppression, Canny hysteresis edge linking, and parametric Hough transform voting.",
        "layers": [
            {
                "algorithm_key": "rest_median",
                "params": {"ksize": 3},
                "auto_convert": True
            },
            {
                "algorithm_key": "seg_canny",
                "params": {"threshold_low": 50, "threshold_high": 150, "aperture_size": 3},
                "auto_convert": True
            },
            {
                "algorithm_key": "seg_hough_lines",
                "params": {"threshold_votes": 70, "min_line_len": 40.0, "max_line_gap": 10.0, "draw_on_source": True},
                "auto_convert": True
            }
        ]
    },
    "noise_restoration": {
        "title": "🔬 Impulse Noise Inoculation & Contraharmonic Filter",
        "description": "Injects Salt & Pepper degradation and restores structural clarity using non-linear Contraharmonic Q filtering.",
        "layers": [
            {
                "algorithm_key": "rest_noise_sp",
                "params": {"salt_prob": 0.04, "pepper_prob": 0.04},
                "auto_convert": True
            },
            {
                "algorithm_key": "rest_contraharmonic",
                "params": {"ksize": 3, "Q": 1.5},
                "auto_convert": True
            },
            {
                "algorithm_key": "rad_hist_eq",
                "params": {},
                "auto_convert": True
            }
        ]
    },
    "hollywood_fourier": {
        "title": "🎥 Hollywood 3D LUT & 2D Fourier Sharpening",
        "description": "Applies cinematic Teal & Orange 3D LUT lattice grading combined with Gaussian high-boost Fourier filtering.",
        "layers": [
            {
                "algorithm_key": "col_3d_lut",
                "params": {"profile": "teal_orange", "strength": 1.0},
                "auto_convert": True
            },
            {
                "algorithm_key": "freq_filter",
                "params": {"filter_type": "gaussian", "pass_type": "lowpass", "cutoff_d0": 45.0, "order_n": 2},
                "auto_convert": True
            },
            {
                "algorithm_key": "seg_high_boost",
                "params": {"boost_a": 1.4, "ksize": 5},
                "auto_convert": True
            }
        ]
    }
}


def init_session_state():
    """Initializes Streamlit state variables if not already present."""
    defaults = {
        "active_layer_stack": [],
        "selected_layer_id": None,
        "history_stack": [],
        "future_stack": [],
        "view_mode": "split",
        "hist_mode": "RGB",
        "selected_domain": "geometry",
        "preview_height": 520,
        "tray_height": 560,
        "tray_collapsed": False
    }
    for key, default_val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_val


def _push_history():
    """Saves deep copy of current layer stack for undo."""
    init_session_state()
    current_copy = copy.deepcopy(st.session_state.get("active_layer_stack", []))
    st.session_state["history_stack"].append(current_copy)
    # Cap history to 15 entries
    if len(st.session_state["history_stack"]) > 15:
        st.session_state["history_stack"].pop(0)
    # Clear future redo on new action
    st.session_state["future_stack"].clear()


def undo():
    """Reverts to previous stack state."""
    init_session_state()
    if st.session_state.get("history_stack"):
        st.session_state["future_stack"].append(copy.deepcopy(st.session_state.get("active_layer_stack", [])))
        st.session_state["active_layer_stack"] = st.session_state["history_stack"].pop()


def redo():
    """Reapplies previously undone stack state."""
    init_session_state()
    if st.session_state.get("future_stack"):
        st.session_state["history_stack"].append(copy.deepcopy(st.session_state.get("active_layer_stack", [])))
        st.session_state["active_layer_stack"] = st.session_state["future_stack"].pop()


def create_layer_dict(algo_key: str, custom_params: Optional[Dict[str, Any]] = None, auto_convert: bool = True) -> Dict[str, Any]:
    """Helper to instantiate a standardized layer dictionary from registry."""
    meta = ALGORITHM_REGISTRY.get(algo_key, {})
    default_params = {}
    for p_name, p_spec in meta.get("params", {}).items():
        default_params[p_name] = p_spec["default"]

    if custom_params:
        default_params.update(custom_params)

    layer_id = f"layer_{uuid.uuid4().hex[:6]}"
    friendly = meta.get("friendly_name", meta.get("name", "Unknown Layer"))
    return {
        "id": layer_id,
        "name": friendly,
        "friendly_name": friendly,
        "algo_name": meta.get("name", "Unknown Algorithm"),
        "simple_description": meta.get("simple_description", meta.get("summary", "")),
        "tooltip": meta.get("tooltip", meta.get("summary", "")),
        "domain": meta.get("domain", "geometry"),
        "algorithm_key": algo_key,
        "enabled": True,
        "auto_convert": auto_convert,
        "params": default_params
    }


def add_layer_to_stack(algo_key: str, custom_params: Optional[Dict[str, Any]] = None, auto_convert: bool = True):
    """Appends an algorithm block to the active stack and selects it."""
    _push_history()
    new_layer = create_layer_dict(algo_key, custom_params=custom_params, auto_convert=auto_convert)
    st.session_state.active_layer_stack.append(new_layer)
    st.session_state.selected_layer_id = new_layer["id"]


def has_crop_layer() -> bool:
    """Returns True if any crop layer is currently in the active pipeline stack."""
    init_session_state()
    for layer in st.session_state.get("active_layer_stack", []):
        if layer.get("algorithm_key") == "geo_shape_crop":
            return True
    return False


def remove_crop_layers_from_stack():
    """
    Removes all shape crop layers from the active pipeline stack,
    completely restoring the full uncropped image canvas scale
    WITHOUT disturbing, resetting, or losing any other active filters.
    """
    init_session_state()
    stack = st.session_state.get("active_layer_stack", [])
    crop_layers = [l for l in stack if l.get("algorithm_key") == "geo_shape_crop"]
    if not crop_layers:
        return

    _push_history()
    st.session_state.active_layer_stack = [
        l for l in stack if l.get("algorithm_key") != "geo_shape_crop"
    ]
    # If the active selection was on a removed crop layer, safely update selection
    removed_ids = {l["id"] for l in crop_layers}
    if st.session_state.get("selected_layer_id") in removed_ids:
        if st.session_state.active_layer_stack:
            st.session_state.selected_layer_id = st.session_state.active_layer_stack[-1]["id"]
            st.session_state["param_active_stage_select"] = st.session_state.selected_layer_id
        else:
            st.session_state.selected_layer_id = None
            st.session_state.pop("param_active_stage_select", None)


def remove_layer(layer_id: str):
    """Removes a layer by ID."""
    _push_history()
    st.session_state.active_layer_stack = [
        layer for layer in st.session_state.active_layer_stack if layer["id"] != layer_id
    ]
    if st.session_state.selected_layer_id == layer_id:
        st.session_state.selected_layer_id = None


def move_layer_up(idx: int):
    """Moves layer at idx one position higher in stack (earlier execution)."""
    if idx > 0 and idx < len(st.session_state.active_layer_stack):
        _push_history()
        stack = st.session_state.active_layer_stack
        stack[idx - 1], stack[idx] = stack[idx], stack[idx - 1]


def move_layer_down(idx: int):
    """Moves layer at idx one position lower in stack (later execution)."""
    if idx >= 0 and idx < len(st.session_state.active_layer_stack) - 1:
        _push_history()
        stack = st.session_state.active_layer_stack
        stack[idx + 1], stack[idx] = stack[idx], stack[idx + 1]


def toggle_layer_enabled(layer_id: str):
    """Toggles bypass state of a layer."""
    _push_history()
    for layer in st.session_state.active_layer_stack:
        if layer["id"] == layer_id:
            layer["enabled"] = not layer["enabled"]
            break


def toggle_layer_autoconvert(layer_id: str):
    """Toggles auto-convert shim for a layer."""
    _push_history()
    for layer in st.session_state.active_layer_stack:
        if layer["id"] == layer_id:
            layer["auto_convert"] = not layer.get("auto_convert", True)
            break


def update_layer_parameter(layer_id: str, param_name: str, value: Any):
    """Updates a single parameter on the target layer."""
    for layer in st.session_state.active_layer_stack:
        if layer["id"] == layer_id:
            layer["params"][param_name] = value
            break


def swap_layer_algorithm(layer_id: str, new_algo_key: str):
    """Hot-swaps target layer's mathematical algorithm in-place."""
    _push_history()
    meta = ALGORITHM_REGISTRY.get(new_algo_key)
    if not meta:
        return

    for layer in st.session_state.active_layer_stack:
        if layer["id"] == layer_id:
            friendly = meta.get("friendly_name", meta.get("name", "Unknown Layer"))
            layer["name"] = friendly
            layer["friendly_name"] = friendly
            layer["algo_name"] = meta.get("name", "Unknown Algorithm")
            layer["simple_description"] = meta.get("simple_description", meta.get("summary", ""))
            layer["tooltip"] = meta.get("tooltip", meta.get("summary", ""))
            layer["domain"] = meta["domain"]
            layer["algorithm_key"] = new_algo_key
            # Populate default parameters for new algorithm
            layer["params"] = {p_k: p_v["default"] for p_k, p_v in meta.get("params", {}).items()}
            break


def clear_layer_stack():
    """Clears all layers in active stack."""
    _push_history()
    st.session_state.active_layer_stack.clear()
    st.session_state.selected_layer_id = None


def load_preset(preset_key: str):
    """Loads a structured industry pipeline preset."""
    preset = PRESETS.get(preset_key)
    if not preset:
        return
    _push_history()
    new_stack = []
    for l_conf in preset["layers"]:
        algo_k = l_conf["algorithm_key"]
        params = l_conf.get("params", {})
        auto_c = l_conf.get("auto_convert", True)
        new_stack.append(create_layer_dict(algo_k, custom_params=params, auto_convert=auto_c))

    st.session_state.active_layer_stack = new_stack
    if new_stack:
        st.session_state.selected_layer_id = new_stack[0]["id"]
