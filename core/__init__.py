"""
ImageLab Pro Core Package
Exports pipeline execution, metrics calculation, and session state manager.
"""

from core.pipeline import execute_pipeline, detect_state, convert_state, PipelineExecutionResult, StageExecutionResult
from core.metrics import calculate_psnr, calculate_mse, calculate_ssim, calculate_entropy, compute_histograms
import core.state_manager as state_mgr

__all__ = [
    "execute_pipeline",
    "detect_state",
    "convert_state",
    "PipelineExecutionResult",
    "StageExecutionResult",
    "calculate_psnr",
    "calculate_mse",
    "calculate_ssim",
    "calculate_entropy",
    "compute_histograms",
    "state_mgr"
]
