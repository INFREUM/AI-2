"""Pydantic models for the AI① HTTP Window envelope.

The raw sensor contract is intentionally permissive for unknown future fields,
while validating the fields that AI② can safely validate today.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.baseline_manager import FEATURES


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number")
    return float(value)


class RawSensor(BaseModel):
    """Known AI① raw-sensor fields; unknown API_SPEC fields are retained."""

    model_config = ConfigDict(extra="allow")

    sample_rate_hz: float | None = None
    unit: str | None = None
    t0_us: int | None = None
    n: int | None = Field(default=None, ge=0)
    axes: dict[str, list[float | None] | None] | None = None
    timestamp_us: list[int | None] | None = None
    quality: Any = None

    @field_validator("sample_rate_hz")
    @classmethod
    def validate_sample_rate(cls, value: float | None) -> float | None:
        if value is None:
            return value
        value = _finite_number(value, "sample_rate_hz")
        if value <= 0:
            raise ValueError("sample_rate_hz must be positive")
        return value

    @model_validator(mode="after")
    def validate_lengths(self) -> "RawSensor":
        if self.n is None:
            return self
        if self.axes is not None:
            for name, values in self.axes.items():
                if values is not None and len(values) != self.n:
                    raise ValueError(f"axes.{name} length must equal n")
        if self.timestamp_us is not None and len(self.timestamp_us) != self.n:
            raise ValueError("timestamp_us length must equal n")
        return self


class RawPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    start_us: int | None = None
    end_us: int | None = None
    sensors: dict[str, RawSensor | None] = Field(default_factory=dict)


class Ai1Condition(BaseModel):
    model_config = ConfigDict(extra="allow")

    condition: str | None = None
    load_level_kg: float
    load_position: str | None = None
    temperature_c: float | None = None

    @field_validator("load_level_kg")
    @classmethod
    def validate_load(cls, value: float) -> float:
        return _finite_number(value, "load_level_kg")

    @field_validator("temperature_c")
    @classmethod
    def validate_temperature(cls, value: float | None) -> float | None:
        return None if value is None else _finite_number(value, "temperature_c")


class Ai1Result(BaseModel):
    model_config = ConfigDict(extra="allow")

    prediction: str | None = None
    warming_up: bool = False
    condition: Ai1Condition
    features: dict[str, Any]

    @model_validator(mode="after")
    def validate_model_c_features(self) -> "Ai1Result":
        missing = [feature for feature in FEATURES if feature not in self.features]
        if missing:
            raise ValueError(f"ai1.features missing Model C feature(s): {missing}")
        for feature in FEATURES:
            self.features[feature] = _finite_number(self.features[feature], f"ai1.features.{feature}")
        return self


class Ai1WindowRequest(BaseModel):
    """The HTTP request accepted at ``POST /api/ai1/windows``."""

    model_config = ConfigDict(extra="allow")

    schema_version: str
    message_type: Literal["ai1_window"]
    bridge_type: Literal["BOX", "TRUSS"]
    experiment_id: str
    window_id: str
    window_start: str
    window_start_us: int
    window_end_us: int
    raw: RawPayload | None
    ai1: Ai1Result
    message_id: str | None = None

    @field_validator("schema_version", "experiment_id", "window_id")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty")
        return value

    @field_validator("window_start_us", "window_end_us")
    @classmethod
    def validate_microseconds(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("Unix microseconds must be an integer")
        return value

    @model_validator(mode="after")
    def validate_window_interval(self) -> "Ai1WindowRequest":
        if self.window_start_us >= self.window_end_us:
            raise ValueError("window_start_us must be earlier than window_end_us")
        return self
