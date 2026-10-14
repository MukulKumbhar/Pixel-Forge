"""
ImageLab Pro - Non-Destructive Layer/Node Pipeline Manager
Executes sequential filter chains over the base raw image with state contract validation,
auto-conversion shims, and per-stage latency telemetry.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from algorithms.registry import ALGORITHM_REGISTRY


def detect_state(image: np.ndarray) -> str:
    """
    Determines image state contract: 'RGB', 'RGBA', 'GRAYSCALE', or 'BINARY'.
    """
    if image is None:
        return "UNKNOWN"

    if image.ndim == 3 and image.shape[2] == 4:
        return "RGBA"
    elif image.ndim == 3 and image.shape[2] == 3:
        return "RGB"
    elif image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        # Single channel: check if values are strictly binary {0, 255}
        unique_vals = np.unique(image)
        if len(unique_vals) <= 2 and np.all(np.isin(unique_vals, [0, 255])):
            return "BINARY"
        return "GRAYSCALE"
    return "UNKNOWN"


def convert_state(image: np.ndarray, source_state: str, target_state: str) -> np.ndarray:
    """
    Auto-conversion shim providing non-destructive format bridges:
    - RGBA -> RGB (drop alpha or alpha blend)
    - RGBA / RGB -> GRAYSCALE (luminance extraction)
    - RGBA / RGB / GRAYSCALE -> BINARY (Otsu optimal thresholding)
    - GRAYSCALE / BINARY -> RGB (channel replication)
    - RGB / GRAYSCALE / BINARY -> RGBA (channel expansion)
    """
    if source_state == target_state:
        return image

    if target_state == "GRAYSCALE":
        if source_state == "RGBA":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        elif source_state == "RGB":
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif source_state == "BINARY":
            return image.copy()  # Binary is already a valid 1-channel format

    elif target_state == "BINARY":
        if source_state == "RGBA":
            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        elif source_state == "RGB":
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        elif source_state == "GRAYSCALE":
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary

    elif target_state == "RGB":
        if source_state == "RGBA":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif source_state in ["GRAYSCALE", "BINARY"]:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif target_state == "RGBA":
        if source_state == "RGB":
            return cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        elif source_state in ["GRAYSCALE", "BINARY"]:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)

    return image


class StageExecutionResult:
    def __init__(
        self,
        layer_id: str,
        layer_name: str,
        domain: str,
        time_ms: float,
        input_state: str,
        output_state: str,
        status: str,
        warning: Optional[str] = None,
        converted: bool = False
    ):
        self.layer_id = layer_id
        self.layer_name = layer_name
        self.domain = domain
        self.time_ms = time_ms
        self.input_state = input_state
        self.output_state = output_state
        self.status = status  # 'executed', 'bypassed', 'conflict_skipped', 'error'
        self.warning = warning
        self.converted = converted


class PipelineExecutionResult:
    def __init__(
        self,
        output_image: np.ndarray,
        stage_results: List[StageExecutionResult],
        total_time_ms: float,
        has_conflicts: bool
    ):
        self.output_image = output_image
        self.stage_results = stage_results
        self.total_time_ms = total_time_ms
        self.has_conflicts = has_conflicts


def execute_pipeline(
    raw_image: np.ndarray,
    layer_stack: List[Dict[str, Any]]
) -> PipelineExecutionResult:
    """
    Executes the active layer stack non-destructively on a fresh copy of raw_image.
    Validates state contracts between each consecutive block.
    """
    if raw_image is None:
        raise ValueError("raw_image cannot be None")

    # Start with pure clone of base frame
    current_frame = raw_image.copy()
    stage_results: List[StageExecutionResult] = []
    has_conflicts = False

    t_start_total = time.perf_counter()

    for idx, layer in enumerate(layer_stack):
        layer_id = layer.get("id", f"layer_{idx}")
        layer_name = layer.get("name", "Unnamed Layer")
        algo_key = layer.get("algorithm_key")
        enabled = layer.get("enabled", True)
        domain = layer.get("domain", "Unknown")
        auto_convert = layer.get("auto_convert", True)
        params = layer.get("params", {})

        # If bypassed, record and carry frame forward
        if not enabled:
            curr_state = detect_state(current_frame)
            stage_results.append(
                StageExecutionResult(
                    layer_id=layer_id,
                    layer_name=layer_name,
                    domain=domain,
                    time_ms=0.0,
                    input_state=curr_state,
                    output_state=curr_state,
                    status="bypassed"
                )
            )
            continue

        algo_meta = ALGORITHM_REGISTRY.get(algo_key)
        if not algo_meta:
            stage_results.append(
                StageExecutionResult(
                    layer_id=layer_id,
                    layer_name=layer_name,
                    domain=domain,
                    time_ms=0.0,
                    input_state=detect_state(current_frame),
                    output_state=detect_state(current_frame),
                    status="error",
                    warning=f"Algorithm key '{algo_key}' not found in registry."
                )
            )
            continue

        expected_contract = algo_meta.get("input_contract", "ANY")
        current_state = detect_state(current_frame)
        was_converted = False
        warning_msg = None

        # State contract validation
        compatible = True
        if expected_contract != "ANY":
            if expected_contract == "GRAYSCALE" and current_state == "BINARY":
                compatible = True  # Binary is acceptable as 1-channel Grayscale
            elif expected_contract != current_state:
                compatible = False

        if not compatible:
            if auto_convert:
                # Apply automatic format bridge
                current_frame = convert_state(current_frame, current_state, expected_contract)
                was_converted = True
                warning_msg = f"Auto-converted upstream tensor from {current_state} to {expected_contract}."
                current_state = expected_contract
            else:
                # Contract violation without auto-convert: raise helpful conflict
                has_conflicts = True
                warning_msg = (
                    f"Format Mismatch: {layer_name} expects {expected_contract}, "
                    f"but received {current_state}. Please enable Auto-Convert or insert a converter stage."
                )
                stage_results.append(
                    StageExecutionResult(
                        layer_id=layer_id,
                        layer_name=layer_name,
                        domain=domain,
                        time_ms=0.0,
                        input_state=current_state,
                        output_state=current_state,
                        status="conflict_skipped",
                        warning=warning_msg
                    )
                )
                continue

        # Execute algorithm with timing
        func = algo_meta["func"]
        t_stage_start = time.perf_counter()
        try:
            # Filter params to those expected by function
            res = func(current_frame, **params)
            t_stage_elapsed = (time.perf_counter() - t_stage_start) * 1000.0

            # Guard against unexpected types
            if isinstance(res, np.ndarray):
                current_frame = np.clip(res, 0, 255).astype(np.uint8)
            else:
                current_frame = np.array(res, dtype=np.uint8)

            output_state = detect_state(current_frame)
            stage_results.append(
                StageExecutionResult(
                    layer_id=layer_id,
                    layer_name=layer_name,
                    domain=domain,
                    time_ms=round(t_stage_elapsed, 2),
                    input_state=current_state,
                    output_state=output_state,
                    status="executed",
                    warning=warning_msg,
                    converted=was_converted
                )
            )
        except Exception as e:
            t_stage_elapsed = (time.perf_counter() - t_stage_start) * 1000.0
            stage_results.append(
                StageExecutionResult(
                    layer_id=layer_id,
                    layer_name=layer_name,
                    domain=domain,
                    time_ms=round(t_stage_elapsed, 2),
                    input_state=current_state,
                    output_state=current_state,
                    status="error",
                    warning=f"Execution error: {str(e)}"
                )
            )

    total_time_ms = round((time.perf_counter() - t_start_total) * 1000.0, 2)
    return PipelineExecutionResult(
        output_image=current_frame,
        stage_results=stage_results,
        total_time_ms=total_time_ms,
        has_conflicts=has_conflicts
    )
