"""Load runtime configuration for the fall prediction pipeline."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

from .predictor import PredictorConfig
from .risk import RiskConfig


RISK_CONFIG_KEYS = {field.name for field in fields(RiskConfig)}
PREDICTOR_CONFIG_KEYS = {
    "baseline_frames",
    "smoothing_window",
    "prefall_consecutive_frames",
    "fall_consecutive_frames",
}


def load_predictor_config(config_path: str | Path) -> PredictorConfig:
    """
    Load a JSON config file into PredictorConfig.

    Supported sections:
    - state_thresholds: prefall_threshold, fall_threshold, min_visibility
    - risk_scoring: RiskConfig feature thresholds and weights
    - temporal_smoothing: PredictorConfig temporal parameters
    """
    data = _load_json_mapping(config_path)

    risk_values: dict[str, Any] = {}
    _copy_known_values(_section(data, "state_thresholds"), RISK_CONFIG_KEYS, risk_values)
    _copy_known_values(_section(data, "risk_scoring"), RISK_CONFIG_KEYS, risk_values)
    _copy_known_values(_section(data, "risk"), RISK_CONFIG_KEYS, risk_values)

    predictor_values: dict[str, Any] = {}
    _copy_known_values(_section(data, "temporal_smoothing"), PREDICTOR_CONFIG_KEYS, predictor_values)
    _copy_known_values(_section(data, "predictor"), PREDICTOR_CONFIG_KEYS, predictor_values)

    return PredictorConfig(risk=RiskConfig(**risk_values), **predictor_values)


def _load_json_mapping(config_path: str | Path) -> Mapping[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, Mapping):
        raise ValueError(f"配置文件顶层必须是 JSON object: {path}")
    return data


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"配置项 {name!r} 必须是 JSON object。")
    return value


def _copy_known_values(
    source: Mapping[str, Any],
    allowed_keys: set[str],
    target: dict[str, Any],
) -> None:
    for key, value in source.items():
        if key in allowed_keys:
            target[key] = value
