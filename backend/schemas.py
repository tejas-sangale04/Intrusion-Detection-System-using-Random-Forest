"""Request and response schemas.

Feature values must be JSON numbers or ``null``. Strings such as ``"12"`` and
booleans are rejected rather than converted: a flow exporter that sends
strings has a bug worth seeing. Which feature *names* are required depends on
the loaded model, so that check happens in ``Detector.to_frame`` and is
reported in the same 422 error format.
"""

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, field_validator

FeatureValue = Union[StrictFloat, StrictInt, None]
Decision = Literal["benign", "anomaly", "known_attack"]

# Dropped before validation, so labelled CSV rows can be sent unchanged.
IGNORED_KEYS = frozenset({"Label"})


def _drop_ignored(flow):
    if isinstance(flow, dict):
        return {k: v for k, v in flow.items() if not (isinstance(k, str) and k.strip() in IGNORED_KEYS)}
    return flow


class FlowIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, FeatureValue] = Field(
        min_length=1, description="CIC-IDS2017 column name -> value; see /model/info for the required names")
    source: str = Field("api", max_length=64, description="free-text origin tag, e.g. 'api' or 'replay'")

    @field_validator("features", mode="before")
    @classmethod
    def _ignore(cls, v):
        return _drop_ignored(v)


class BatchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    flows: list[dict[str, FeatureValue]] = Field(min_length=1)
    source: str = Field("api", max_length=64)

    @field_validator("flows", mode="before")
    @classmethod
    def _ignore(cls, v):
        return [_drop_ignored(f) for f in v] if isinstance(v, list) else v


class PredictionOut(BaseModel):
    prediction_id: str
    created_at: str
    source: str
    decision: Decision
    attack_probability: float = Field(description="Random Forest probability that the flow is an attack")
    anomaly_score: float = Field(description="Isolation Forest score; higher = more unusual than benign training traffic")
    classifier_threshold: float
    anomaly_threshold: float
    missing_values: list[str] = Field(description="features sent as null or infinite, filled with the training median")
    explanation_url: str


class BatchOut(BaseModel):
    count: int
    decision_counts: dict[str, int]
    predictions: list[PredictionOut]


class Contribution(BaseModel):
    feature: str
    value: float
    shap: float


class ExplanationOut(BaseModel):
    prediction_id: str
    decision: Decision
    attack_probability: float
    base_value: float
    top_features: list[Contribution]
    summary: str
    scope: str


class AlertOut(BaseModel):
    prediction_id: str
    created_at: str
    source: str
    decision: Decision
    attack_probability: float
    anomaly_score: float
    missing_values: list[str]


class AlertsOut(BaseModel):
    count: int
    alerts: list[AlertOut]
    note: str
