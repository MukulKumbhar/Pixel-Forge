"""
Pixel-Forge - Parameter Adjustment Drawer & Filter Catalog Studio
Provides:
1. Filter Catalog browser (Select Domain & Select Algorithm with live preview and 1-click addition).
2. Parameter drawer (stage switcher, micro-info card, hot-swapping, auto-convert, synchronized sliders & steppers).
"""

from typing import Optional
import streamlit as st
from algorithms.registry import ALGORITHM_REGISTRY, DOMAINS
from core.state_manager import (
    _push_history,
    add_layer_to_stack,
    remove_layer,
    swap_layer_algorithm,
    update_layer_parameter,
)


def render_filter_catalog():
    """
    Renders the Domain & Algorithm Catalog browser.
    Allows user to select a domain, choose an algorithm, inspect its mathematical summary & real-world example,
    and add it to the active pipeline stack.
    """
    domain_keys = list(DOMAINS.keys())
    col_domain, col_algo = st.columns([1, 1.4])

    with col_domain:
        sel_domain = st.selectbox(
            "Select Domain / Category",
            options=domain_keys,
            format_func=lambda d: DOMAINS[d].get("title", DOMAINS[d].get("name", str(d))),
            key="catalog_sel_domain"
        )
        domain_info = DOMAINS.get(sel_domain, {})
        st.caption(f"📁 **{domain_info.get('unit', '')}**: {domain_info.get('description', '')}")

    with col_algo:
        domain_algos = {
            k: f"{v['name']} ({v.get('input_contract', 'ANY')} → {v.get('output_contract', 'SAME')})"
            for k, v in ALGORITHM_REGISTRY.items()
            if v.get("domain") == sel_domain
        }
        sel_algo = st.selectbox(
            "Select Algorithm",
            options=list(domain_algos.keys()),
            format_func=lambda k: domain_algos[k],
            key="catalog_sel_algo"
        )

    # Algorithm detail micro-card
    meta = ALGORITHM_REGISTRY.get(sel_algo, {})
    summary = meta.get("summary", "")
    example = meta.get("example", "")
    in_contract = meta.get("input_contract", "ANY")
    out_contract = meta.get("output_contract", "SAME")

    detail_col1, detail_col2 = st.columns([3, 1.2])
    with detail_col1:
        example_html = f"<div style='margin-top: 4px;'><span class='example-title'>🎯 Practical Example:</span> {example}</div>" if example else ""
        st.markdown(
            f"""
            <div class="filter-micro-info" style="margin-bottom: 0.6rem;">
                <div><span class="info-title">💡 What it does:</span> {summary}</div>
                {example_html}
            </div>
            """,
            unsafe_allow_html=True
        )

    with detail_col2:
        st.markdown(
            f"""
            <div style="padding: 0.6rem 0.8rem; background: rgba(30, 41, 59, 0.45); border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 8px;">
                <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">CONTRACT SPEC</div>
                <div style="margin-top: 4px;">
                    <span class="badge badge-unit">In: {in_contract}</span>
                    <span class="badge badge-rgb" style="margin-left: 4px;">Out: {out_contract}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("➕ Add to Pipeline Stack", key=f"btn_add_algo_{sel_algo}", use_container_width=True, type="primary"):
            add_layer_to_stack(sel_algo)
            st.rerun()


def render_parameter_drawer(selected_layer_id: Optional[str] = None):
    """
    Renders active filter parameter controls and fine-tuning.
    Provides stage selector, algorithm hot-swapping, auto-convert toggling,
    and synchronized sliders and steppers.
    """
    stack = st.session_state.get("active_layer_stack", [])
    if not stack:
        st.info("Pipeline layer stack is currently empty. Open the **Filter Catalog** above to add a stage, or load a preset in the sidebar.")
        return

    # Ensure a valid selected_layer_id
    layer_ids = [l["id"] for l in stack]
    if not selected_layer_id or selected_layer_id not in layer_ids:
        selected_layer_id = stack[-1]["id"]
        st.session_state.selected_layer_id = selected_layer_id

    # Layer selector to quickly switch which stage to edit
    layer_map = {l["id"]: f"Stage {i+1}: {l['name']}" for i, l in enumerate(stack)}

    col_sel, col_bridge, col_reset = st.columns([2.5, 1.3, 1])
    with col_sel:
        chosen_layer_id = st.selectbox(
            "Select Active Stage to Edit",
            options=layer_ids,
            format_func=lambda lid: layer_map.get(lid, lid),
            index=layer_ids.index(selected_layer_id),
            key="param_active_stage_select"
        )
        if chosen_layer_id != selected_layer_id:
            st.session_state.selected_layer_id = chosen_layer_id
            st.rerun()

    # Find active layer
    layer = next((l for l in stack if l["id"] == st.session_state.selected_layer_id), stack[-1])
    algo_key = layer["algorithm_key"]
    meta = ALGORITHM_REGISTRY.get(algo_key, {})
    domain_key = meta.get("domain", "geometry")
    domain_info = DOMAINS.get(domain_key, {})

    with col_bridge:
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

    with col_reset:
        st.write("")
        if st.button("↺ Reset Stage", key=f"reset_stage_{layer['id']}", use_container_width=True, help="Reset this stage to default parameters"):
            _push_history()
            param_specs = meta.get("params", {})
            layer["params"] = {p_k: p_v["default"] for p_k, p_v in param_specs.items()}
            st.rerun()

    # Micro-info card for active filter
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

    # Hot-swapping row
    domain_algos = {
        k: v["name"] for k, v in ALGORITHM_REGISTRY.items() if v.get("domain") == domain_key
    }
    if algo_key in domain_algos:
        hs_col1, hs_col2 = st.columns([2, 1])
        with hs_col1:
            new_algo = st.selectbox(
                "Hot-Swap Algorithm within Domain",
                options=list(domain_algos.keys()),
                format_func=lambda k: domain_algos[k],
                index=list(domain_algos.keys()).index(algo_key),
                key=f"hotswap_{layer['id']}"
            )
            if new_algo != algo_key:
                swap_layer_algorithm(layer["id"], new_algo)
                st.rerun()
        with hs_col2:
            st.markdown(
                f"""
                <div style="margin-top: 24px;">
                    <span class="badge badge-unit">{domain_info.get('unit', '')}</span>
                    <span class="badge badge-rgb" style="margin-left: 4px;">In: {meta.get('input_contract', 'ANY')}</span>
                    <span class="badge badge-gray" style="margin-left: 4px;">Out: {meta.get('output_contract', 'SAME')}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    # Sliders and numeric steppers
    params = layer.get("params", {})
    param_specs = meta.get("params", {})

    if not param_specs:
        st.info("ℹ️ This mathematical point operation has no adjustable parameters.")
    else:
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


def _render_param_input(layer_id: str, p_name: str, p_spec: dict, current_val):
    """Renders synchronized parameter control."""
    p_type = p_spec.get("type", "float")
    label = p_spec.get("label", p_name)

    if p_type == "float":
        min_v = float(p_spec.get("min", 0.0))
        max_v = float(p_spec.get("max", 1.0))
        step_v = float(p_spec.get("step", 0.05))

        st.markdown(f"<div style='font-size: 0.92rem; font-weight: 600; color: #f1f5f9; margin-bottom: 4px;'>{label}</div>", unsafe_allow_html=True)
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
            st.rerun()

    elif p_type == "int":
        min_v = int(p_spec.get("min", 0))
        max_v = int(p_spec.get("max", 100))
        step_v = int(p_spec.get("step", 1))

        st.markdown(f"<div style='font-size: 0.92rem; font-weight: 600; color: #f1f5f9; margin-bottom: 4px;'>{label}</div>", unsafe_allow_html=True)
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
            # Enforce odd requirement if specified
            if p_spec.get("odd", False) and new_val % 2 == 0:
                new_val += 1
            update_layer_parameter(layer_id, p_name, new_val)
            st.rerun()

    elif p_type == "choice":
        options = p_spec.get("options", [])
        st.markdown(f"<div style='font-size: 0.92rem; font-weight: 600; color: #f1f5f9; margin-bottom: 4px;'>{label}</div>", unsafe_allow_html=True)
        idx = options.index(current_val) if current_val in options else 0
        new_choice = st.selectbox(
            label,
            options=options,
            index=idx,
            key=f"choice_{layer_id}_{p_name}",
            label_visibility="collapsed"
        )
        if new_choice != current_val:
            update_layer_parameter(layer_id, p_name, new_choice)
            st.rerun()

    elif p_type == "bool":
        new_bool = st.checkbox(
            label,
            value=bool(current_val),
            key=f"bool_{layer_id}_{p_name}"
        )
        if new_bool != current_val:
            update_layer_parameter(layer_id, p_name, new_bool)
            st.rerun()
