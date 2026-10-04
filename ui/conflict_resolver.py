"""
ImageLab Pro - State Contract Conflict Resolver & Prerequisite Alerts
Detects tensor type mismatches (RGB vs Grayscale vs Binary) across sequential layers
and provides user-friendly 1-click resolution actions.
"""

from typing import List
import streamlit as st
from core.pipeline import StageExecutionResult
from core.state_manager import create_layer_dict, _push_history


def render_conflict_alerts(stage_results: List[StageExecutionResult]):
    """
    Renders contextual prerequisite warnings and 1-click auto-fix actions
    for any layers with state contract violations.
    """
    conflicts = [r for r in stage_results if r.status == "conflict_skipped"]
    conversions = [r for r in stage_results if r.converted]

    # Show informative notices for auto-converted layers
    if conversions:
        for c in conversions:
            st.info(f"⚡ **Auto-Bridge Active:** `{c.layer_name}` automatically converted upstream input from **{c.input_state}** to **{c.output_state}**.")

    if not conflicts:
        st.session_state.pop("has_shown_conflict_toast", None)
        return

    # Trigger friendly toast once per conflict state
    if not st.session_state.get("has_shown_conflict_toast", False):
        st.session_state.has_shown_conflict_toast = True
        c0 = conflicts[0]
        st.toast(
            f"⚠️ Incompatible Input Format on '{c0.layer_name}': Expected {c0.output_state}, received {c0.input_state}. Enable Auto-Convert or insert a converter.",
            icon="⚠️"
        )

    # Render actionable prerequisite alert cards
    for conflict in conflicts:
        with st.container():
            st.markdown(
                f"""
                <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 8px; padding: 1rem; margin-bottom: 0.8rem;">
                    <div style="display: flex; align-items: center; gap: 8px; color: #f87171; font-weight: 600; font-size: 0.95rem; margin-bottom: 4px;">
                        <span>⚠️ Pipeline Prerequisite Required</span>
                    </div>
                    <p style="font-size: 0.88rem; color: #cbd5e1; margin: 4px 0 8px 0;">
                        Layer <strong>{conflict.layer_name}</strong> received <strong>{conflict.input_state}</strong>, but requires a single-channel <strong>{conflict.output_state}</strong> input tensor.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            col_btn1, col_btn2 = st.columns([1, 1])

            # Button 1: Toggle Auto-Convert for this layer
            with col_btn1:
                if st.button(f"⚡ Enable Auto-Convert for '{conflict.layer_name}'", key=f"fix_auto_{conflict.layer_id}"):
                    _push_history()
                    for l in st.session_state.active_layer_stack:
                        if l["id"] == conflict.layer_id:
                            l["auto_convert"] = True
                            break
                    st.rerun()

            # Button 2: Insert Grayscale & Otsu Threshold before this layer
            with col_btn2:
                if st.button(f"➕ Insert Grayscale & Otsu Before Layer", key=f"fix_insert_{conflict.layer_id}"):
                    _push_history()
                    target_idx = 0
                    for idx, l in enumerate(st.session_state.active_layer_stack):
                        if l["id"] == conflict.layer_id:
                            target_idx = idx
                            break

                    otsu_layer = create_layer_dict("seg_otsu", auto_convert=True)
                    st.session_state.active_layer_stack.insert(target_idx, otsu_layer)
                    st.rerun()
