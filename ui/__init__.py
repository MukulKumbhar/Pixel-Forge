"""
ImageLab Pro UI Package
Exports studio styling, viewport canvas, parameter drawer, category dock, and conflict alerts.
"""

from ui.styles import apply_studio_styles
from ui.viewport import render_dual_viewport, render_telemetry_banner, render_histogram_dock
from ui.parameter_drawer import render_parameter_drawer
from ui.bottom_tray import render_bottom_tray
from ui.conflict_resolver import render_conflict_alerts

__all__ = [
    "apply_studio_styles",
    "render_dual_viewport",
    "render_telemetry_banner",
    "render_histogram_dock",
    "render_parameter_drawer",
    "render_bottom_tray",
    "render_conflict_alerts"
]
