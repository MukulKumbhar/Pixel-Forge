"""
Pixel-Forge - Parameter Adjustment Drawer & Filter Catalog Studio
Provides:
1. Filter Catalog browser (Select Domain & Select Algorithm with live preview and 1-click addition).
2. Parameter drawer (stage switcher, micro-info card, hot-swapping, auto-convert, high-precision sliders).
All controls use native Streamlit on_change event callbacks to guarantee zero-flicker, loop-free execution.
"""

from typing import Any, Optional
import streamlit as st
from algorithms.registry import ALGORITHM_REGISTRY, DOMAINS
from core.state_manager import (
    _push_history,
    add_layer_to_stack,
    remove_layer,
    swap_layer_algorithm,
    update_layer_parameter,
)


def _on_catalog_domain_change():
    """Synchronizes algorithm dropdown when the user changes domain."""
    domain = st.session_state.get("catalog_sel_domain")
    if domain:
        domain_algos = [k for k, v in ALGORITHM_REGISTRY.items() if v.get("domain") == domain]
        if domain_algos:
            st.session_state["catalog_sel_algo"] = domain_algos[0]


def _on_stage_select():
    """Synchronizes selected_layer_id when the user changes stage in the dropdown."""
    chosen = st.session_state.get("param_active_stage_select")
    if chosen:
        st.session_state.selected_layer_id = chosen


def _on_autoconvert_change(layer_id: str, widget_key: str):
    """Toggles auto-convert shim for a layer without manual rerun."""
    new_val = st.session_state.get(widget_key, True)
    _push_history()
    for layer in st.session_state.get("active_layer_stack", []):
        if layer["id"] == layer_id:
            layer["auto_convert"] = new_val
            break


def _on_hotswap_change(layer_id: str, widget_key: str):
    """Hot-swaps target layer algorithm in-place without manual rerun."""
    new_algo = st.session_state.get(widget_key)
    if new_algo:
        swap_layer_algorithm(layer_id, new_algo)
        # Update widget session keys for the new algorithm's defaults
        meta = ALGORITHM_REGISTRY.get(new_algo, {})
        for p_k, p_v in meta.get("params", {}).items():
            w_key = f"param_{layer_id}_{p_k}"
            st.session_state[w_key] = p_v["default"]


def _on_param_change(layer_id: str, p_name: str, widget_key: str, p_type: str = "float", odd: bool = False):
    """
    Native Streamlit on_change callback.
    Executes BEFORE the script reruns. Directly updates the in-memory parameter
    on the target layer without ever invoking st.rerun().
    """
    if widget_key not in st.session_state:
        return
    val = st.session_state[widget_key]
    if p_type == "int":
        val = int(val)
        if odd and val % 2 == 0:
            val += 1
            st.session_state[widget_key] = val
    elif p_type == "float":
        val = float(val)
    update_layer_parameter(layer_id, p_name, val)


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
            key="catalog_sel_domain",
            on_change=_on_catalog_domain_change
        )
        domain_info = DOMAINS.get(sel_domain, {})
        st.caption(f"📁 **{domain_info.get('unit', '')}**: {domain_info.get('description', '')}")

    with col_algo:
        domain_algos = {
            k: v.get("friendly_name", v["name"])
            for k, v in ALGORITHM_REGISTRY.items()
            if v.get("domain") == sel_domain
        }
        domain_algo_keys = list(domain_algos.keys())
        if st.session_state.get("catalog_sel_algo") not in domain_algo_keys and domain_algo_keys:
            st.session_state["catalog_sel_algo"] = domain_algo_keys[0]

        cur_selected_algo = st.session_state.get("catalog_sel_algo", domain_algo_keys[0] if domain_algo_keys else None)
        cur_algo_meta = ALGORITHM_REGISTRY.get(cur_selected_algo, {}) if cur_selected_algo else {}
        algo_tooltip = cur_algo_meta.get("tooltip", cur_algo_meta.get("simple_description", "Choose a filter to add to the pipeline."))

        sel_algo = st.selectbox(
            "Choose Filter / Tool",
            options=domain_algo_keys,
            format_func=lambda k: domain_algos[k],
            key="catalog_sel_algo",
            help=algo_tooltip
        )

    # Algorithm detail micro-card
    meta = ALGORITHM_REGISTRY.get(sel_algo, {})
    friendly_name = meta.get("friendly_name", meta.get("name", ""))
    academic_name = meta.get("name", "")
    simple_desc = meta.get("simple_description", meta.get("summary", ""))
    example = meta.get("example", "")
    in_contract = meta.get("input_contract", "ANY")
    out_contract = meta.get("output_contract", "SAME")
    formula = meta.get("formula", "")
    viva_theory = meta.get("viva_theory", meta.get("summary", ""))

    detail_col1, detail_col2 = st.columns([3, 1.2])
    with detail_col1:
        example_html = f"<div style='margin-top: 5px;'><span class='example-title'>🎯 Real-World Use:</span> <span style='color: #cbd5e1;'>{example}</span></div>" if example else ""
        st.markdown(
            f"""
            <div class="filter-micro-info" style="margin-bottom: 0.6rem;">
                <div><span class="info-title">💡 What this filter does:</span> <span style="color: #f1f5f9; font-weight: 500;">{simple_desc}</span></div>
                {example_html}
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander(f"🔬 **Academic Algorithm Details: {academic_name}**", expanded=False):
            st.markdown(f"**Academic Title:** `{academic_name}` • **Domain:** `{meta.get('domain', '').capitalize()}`")
            if formula:
                st.markdown("**Mathematical Model:**")
                st.latex(formula)
            if viva_theory:
                st.markdown(f"**DIP Syllabus / Viva Notes:**\n> {viva_theory}")

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
            # Sync stage selector to newly added layer
            if st.session_state.active_layer_stack:
                new_id = st.session_state.active_layer_stack[-1]["id"]
                st.session_state.selected_layer_id = new_id
                st.session_state["param_active_stage_select"] = new_id
            st.rerun()


def render_parameter_drawer(selected_layer_id: Optional[str] = None):
    """
    Renders active filter parameter controls and fine-tuning.
    Provides stage selector, algorithm hot-swapping, auto-convert toggling,
    and high-precision parameter sliders.
    All inputs use on_change callbacks with zero loop/rerun jitter.
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

    # Strictly keep the selectbox widget key synchronized with selected_layer_id
    if st.session_state.get("param_active_stage_select") not in layer_ids or st.session_state.get("param_active_stage_select") != selected_layer_id:
        st.session_state["param_active_stage_select"] = selected_layer_id

    layer_map = {l["id"]: f"Stage {i+1}: {l.get('friendly_name', l['name'])}" for i, l in enumerate(stack)}

    col_sel, col_bridge, col_reset = st.columns([2.5, 1.3, 1])
    with col_sel:
        st.selectbox(
            "Select Active Stage to Edit",
            options=layer_ids,
            format_func=lambda lid: layer_map.get(lid, lid),
            key="param_active_stage_select",
            on_change=_on_stage_select,
            help="Switch between active stages to tune sliders or reset parameters."
        )

    # Find active layer
    layer = next((l for l in stack if l["id"] == st.session_state.selected_layer_id), stack[-1])
    algo_key = layer["algorithm_key"]
    meta = ALGORITHM_REGISTRY.get(algo_key, {})
    domain_key = meta.get("domain", "geometry")
    domain_info = DOMAINS.get(domain_key, {})

    with col_bridge:
        ac_key = f"autoconvert_{layer['id']}"
        if ac_key not in st.session_state:
            st.session_state[ac_key] = layer.get("auto_convert", True)
        st.toggle(
            "Auto-Convert Shims",
            key=ac_key,
            on_change=_on_autoconvert_change,
            args=(layer["id"], ac_key),
            help="Automatically bridges tensor format mismatches (e.g. RGB to Grayscale/Binary)"
        )

    with col_reset:
        st.write("")
        if st.button("↺ Reset Stage", key=f"reset_stage_{layer['id']}", use_container_width=True, help="Reset this stage to default parameters"):
            _push_history()
            param_specs = meta.get("params", {})
            layer["params"] = {p_k: p_v["default"] for p_k, p_v in param_specs.items()}
            # Reconcile widget keys so UI reflects defaults immediately
            for p_k, p_v in param_specs.items():
                st.session_state[f"param_{layer['id']}_{p_k}"] = p_v["default"]
            st.rerun()

    # Micro-info card for active filter
    simple_desc = meta.get("simple_description", meta.get("summary", ""))
    example = meta.get("example", "")
    academic_name = meta.get("name", "")
    formula = meta.get("formula", "")
    viva_theory = meta.get("viva_theory", meta.get("summary", ""))

    if simple_desc or example:
        example_html = f"<div style='margin-top: 3px;'><span class='example-title'>🎯 Real-World Use:</span> <span style='color: #cbd5e1;'>{example}</span></div>" if example else ""
        st.markdown(
            f"""
            <div class="filter-micro-info">
                <div><span class="info-title">💡 What it does:</span> <span style="color: #f1f5f9; font-weight: 500;">{simple_desc}</span></div>
                {example_html}
            </div>
            """,
            unsafe_allow_html=True
        )
        with st.expander(f"🔬 **Academic Algorithm Info: {academic_name}**", expanded=False):
            st.markdown(f"**Academic Reference:** `{academic_name}`")
            if formula:
                st.markdown("**Mathematical Formulation:**")
                st.latex(formula)
            if viva_theory:
                st.markdown(f"**Theory & Concept:**\n> {viva_theory}")

    # Hot-swapping row
    domain_algos = {
        k: v.get("friendly_name", v["name"]) for k, v in ALGORITHM_REGISTRY.items() if v.get("domain") == domain_key
    }
    if algo_key in domain_algos:
        hs_col1, hs_col2 = st.columns([2, 1])
        with hs_col1:
            hs_key = f"hotswap_{layer['id']}"
            if hs_key not in st.session_state or st.session_state[hs_key] not in domain_algos:
                st.session_state[hs_key] = algo_key
            st.selectbox(
                "Hot-Swap with Another Filter in this Category",
                options=list(domain_algos.keys()),
                format_func=lambda k: domain_algos[k],
                key=hs_key,
                on_change=_on_hotswap_change,
                args=(layer["id"], hs_key),
                help="Swap this stage with any other filter in the same category without losing your place in the pipeline."
            )
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

    # Sliders and numeric parameter controls
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


def _render_param_input(layer_id: str, p_name: str, p_spec: dict, current_val: Any):
    """
    Renders high-precision parameter control using native on_change callback.
    Guarantees zero-loop, zero-flicker updating.
    """
    p_type = p_spec.get("type", "float")
    label = p_spec.get("label", p_name)
    w_key = f"param_{layer_id}_{p_name}"

    if p_type == "float":
        min_v = float(p_spec.get("min", 0.0))
        max_v = float(p_spec.get("max", 1.0))
        step_v = float(p_spec.get("step", 0.05))

        cur_f = float(current_val)
        if cur_f < min_v:
            cur_f = min_v
        if cur_f > max_v:
            cur_f = max_v

        if w_key not in st.session_state:
            st.session_state[w_key] = cur_f

        st.slider(
            label=f"{label} ({p_name})",
            min_value=min_v,
            max_value=max_v,
            step=step_v,
            key=w_key,
            on_change=_on_param_change,
            args=(layer_id, p_name, w_key, "float", False),
            help=f"Range: [{min_v}, {max_v}], Default: {p_spec.get('default')}"
        )

    elif p_type == "int":
        min_v = int(p_spec.get("min", 0))
        max_v = int(p_spec.get("max", 100))
        step_v = int(p_spec.get("step", 1))
        odd_req = bool(p_spec.get("odd", False))

        cur_i = int(current_val)
        if cur_i < min_v:
            cur_i = min_v
        if cur_i > max_v:
            cur_i = max_v

        if w_key not in st.session_state:
            st.session_state[w_key] = cur_i

        st.slider(
            label=f"{label} ({p_name})",
            min_value=min_v,
            max_value=max_v,
            step=step_v,
            key=w_key,
            on_change=_on_param_change,
            args=(layer_id, p_name, w_key, "int", odd_req),
            help=f"Range: [{min_v}, {max_v}], Default: {p_spec.get('default')}"
        )

    elif p_type in ("select", "choice"):
        options = p_spec.get("options", [])
        idx = options.index(current_val) if current_val in options else 0

        if w_key not in st.session_state or st.session_state[w_key] not in options:
            st.session_state[w_key] = options[idx] if options else ""

        st.selectbox(
            label=f"{label} ({p_name})",
            options=options,
            key=w_key,
            on_change=_on_param_change,
            args=(layer_id, p_name, w_key, "select", False),
            help=f"Default: {p_spec.get('default')}"
        )

    elif p_type == "bool":
        if w_key not in st.session_state:
            st.session_state[w_key] = bool(current_val)

        st.checkbox(
            label=f"{label} ({p_name})",
            key=w_key,
            on_change=_on_param_change,
            args=(layer_id, p_name, w_key, "bool", False),
            help=f"Default: {p_spec.get('default')}"
        )
