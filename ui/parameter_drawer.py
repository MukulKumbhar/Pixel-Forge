"""
ImageLab Pro - Parameter Adjustment Drawer & Fine-Grained Channel Studio
Provides synchronized slider and numeric controls, algorithm hot-swapping,
auto-convert toggling, and per-stage parameter resets.
"""

from typing import Optional
import streamlit as st
from algorithms.registry import ALGORITHM_REGISTRY, DOMAINS
from core.state_manager import (
    _push_history,
    swap_layer_algorithm,
    update_layer_parameter,
    remove_layer
)


def render_parameter_drawer(selected_layer_id: Optional[str]):
    """
    Renders parameter adjustment drawer when a layer is selected for editing.
    """
    if not selected_layer_id:
        st.markdown(
            """
            <div style="text-align: center; padding: 2.5rem 1rem; color: #94a3b8;">
                <span style="font-size: 2.2rem;">⚙️</span>
                <div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-top: 6px;">No Filter Stage Selected for Editing</div>
                <div style="font-size: 0.88rem; color: #94a3b8; margin-top: 4px; max-width: 600px; margin-left: auto; margin-right: auto; line-height: 1.5;">
                    Click the <b>⚙️ Gear icon</b> on any stage in the <b>Active Layer Stack</b> (sidebar), or click <b>➕ Add to Pipeline</b> in the <b>Browse Filter Catalog</b> tab to inspect and tweak parameters here.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # Find layer in active stack
    layer_idx = None
    layer = None
    for idx, l in enumerate(st.session_state.active_layer_stack):
        if l["id"] == selected_layer_id:
            layer_idx = idx
            layer = l
            break

    if layer is None or layer_idx is None:
        st.session_state.selected_layer_id = None
        return

    algo_key = layer["algorithm_key"]
    meta = ALGORITHM_REGISTRY.get(algo_key, {})
    domain_key = meta.get("domain", "geometry")
    domain_info = DOMAINS.get(domain_key, {})

    with st.container():
        st.markdown(
            f"""
            <div class="param-drawer">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                    <div>
                        <span style="font-size: 1.05rem; font-weight: 600; color: #f3f4f6;">
                            ⚙️ Stage {layer_idx + 1}: {layer['name']}
                        </span>
                        <span class="badge badge-unit" style="margin-left: 8px;">{domain_info.get('unit', '')}</span>
                        <span class="badge badge-rgb" style="margin-left: 4px;">In: {meta.get('input_contract', 'ANY')}</span>
                        <span class="badge badge-gray" style="margin-left: 4px;">Out: {meta.get('output_contract', 'SAME')}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Compact Filter Micro-Info Card (Replaces verbose Viva Guide)
        summary = meta.get("summary", "")
        example = meta.get("example", "")
        if summary or example:
            example_html = f"<div style='margin-top: 3px;'><span class='example-title'>🎯 Example:</span> {example}</div>" if example else ""
            st.markdown(
                f"""
                <div class="filter-micro-info">
                    <div><span class="info-title">💡 What it does:</span> {summary}</div>
                    {example_html}
                </div>
                """,
                unsafe_allow_html=True
            )

        # Hot-swapping dropdown & auto-convert toggle
        top_ctrl1, top_ctrl2, top_ctrl3 = st.columns([2, 1.5, 1])
        with top_ctrl1:
            # Filter algorithms matching current domain for hot-swapping
            domain_algos = {
                k: v["name"] for k, v in ALGORITHM_REGISTRY.items() if v.get("domain") == domain_key
            }
            if algo_key in domain_algos:
                new_algo = st.selectbox(
                    "Hot-Swap Algorithm",
                    options=list(domain_algos.keys()),
                    format_func=lambda k: domain_algos[k],
                    index=list(domain_algos.keys()).index(algo_key),
                    key=f"hotswap_{layer['id']}"
                )
                if new_algo != algo_key:
                    swap_layer_algorithm(layer["id"], new_algo)
                    st.rerun()

        with top_ctrl2:
            auto_c = st.toggle(
                "Auto-Convert Shims",
                value=layer.get("auto_convert", True),
                key=f"autoconvert_{layer['id']}",
                help="Automatically bridges tensor format mismatches (e.g. RGB to Grayscale/Binary)"
            )
            if auto_c != layer.get("auto_convert", True):
                _push_history()
                layer["auto_convert"] = auto_c
                st.rerun()

        with top_ctrl3:
            if st.button("✖ Close Drawer", key=f"close_drawer_{layer['id']}"):
                st.session_state.selected_layer_id = None
                st.rerun()

        st.divider()

        # Render individual parameters
        params = layer.get("params", {})
        param_specs = meta.get("params", {})

        if not param_specs:
            st.info("ℹ️ This mathematical point operation has no adjustable parameters.")
        else:
            # Group parameters in 2-column grid
            param_items = list(param_specs.items())
            for i in range(0, len(param_items), 2):
                col_a, col_b = st.columns(2)
                p1_name, p1_spec = param_items[i]

                with col_a:
                    _render_param_input(layer["id"], p1_name, p1_spec, params.get(p1_name, p1_spec["default"]))

                if i + 1 < len(param_items):
                    p2_name, p2_spec = param_items[i + 1]
                    with col_b:
                        _render_param_input(layer["id"], p2_name, p2_spec, params.get(p2_name, p2_spec["default"]))

        # Action buttons row: Reset stage & Delete stage
        st.divider()
        b_col1, b_col2, b_col3 = st.columns([1, 1, 2])
        with b_col1:
            if st.button("↺ Reset to Defaults", key=f"reset_stage_{layer['id']}"):
                _push_history()
                layer["params"] = {p_k: p_v["default"] for p_k, p_v in param_specs.items()}
                st.rerun()

        with b_col2:
            if st.button("🗑️ Delete Stage", key=f"delete_stage_{layer['id']}", type="secondary"):
                remove_layer(layer["id"])
                st.session_state.selected_layer_id = None
                st.rerun()


def _render_param_input(layer_id: str, p_name: str, p_spec: dict, current_val):
    """Renders synchronized parameter control."""
    p_type = p_spec.get("type", "float")
    label = p_spec.get("label", p_name)

    if p_type == "float":
        min_v = float(p_spec.get("min", 0.0))
        max_v = float(p_spec.get("max", 1.0))
        step_v = float(p_spec.get("step", 0.05))

        st.markdown(f"<div style='font-size: 0.94rem; font-weight: 600; color: #f1f5f9; margin-bottom: 4px;'>{label}</div>", unsafe_allow_html=True)
        col_sl, col_num = st.columns([3, 1.2])
        with col_sl:
            v_sl = st.slider(
                label,
                min_value=min_v,
                max_value=max_v,
                value=float(current_val),
                step=step_v,
                key=f"sl_{layer_id}_{p_name}",
                label_visibility="collapsed"
            )
        with col_num:
            v_num = st.number_input(
                label,
                min_value=min_v,
                max_value=max_v,
                value=float(current_val),
                step=step_v,
                key=f"num_{layer_id}_{p_name}",
                label_visibility="collapsed"
            )

        new_val = current_val
        if abs(v_sl - current_val) > 1e-6:
            new_val = v_sl
        elif abs(v_num - current_val) > 1e-6:
            new_val = v_num

        if abs(new_val - current_val) > 1e-6:
            update_layer_parameter(layer_id, p_name, new_val)

    elif p_type == "int":
        min_v = int(p_spec.get("min", 0))
        max_v = int(p_spec.get("max", 100))
        step_v = int(p_spec.get("step", 1))

        st.markdown(f"<div style='font-size: 0.94rem; font-weight: 600; color: #f1f5f9; margin-bottom: 4px;'>{label}</div>", unsafe_allow_html=True)
        col_sl, col_num = st.columns([3, 1.2])
        with col_sl:
            v_sl = st.slider(
                label,
                min_value=min_v,
                max_value=max_v,
                value=int(current_val),
                step=step_v,
                key=f"sl_{layer_id}_{p_name}",
                label_visibility="collapsed"
            )
        with col_num:
            v_num = st.number_input(
                label,
                min_value=min_v,
                max_value=max_v,
                value=int(current_val),
                step=step_v,
                key=f"num_{layer_id}_{p_name}",
                label_visibility="collapsed"
            )

        new_val = current_val
        if v_sl != current_val:
            new_val = v_sl
        elif v_num != current_val:
            new_val = v_num

        if new_val != current_val:
            update_layer_parameter(layer_id, p_name, new_val)

    elif p_type == "select":
        options = p_spec.get("options", [])
        idx = options.index(current_val) if current_val in options else 0
        val = st.selectbox(
            label,
            options=options,
            index=idx,
            key=f"select_{layer_id}_{p_name}"
        )
        if val != current_val:
            update_layer_parameter(layer_id, p_name, val)

    elif p_type == "bool":
        val = st.checkbox(
            label,
            value=bool(current_val),
            key=f"check_{layer_id}_{p_name}"
        )
        if val != current_val:
            update_layer_parameter(layer_id, p_name, val)
