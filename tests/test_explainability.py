from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.explainability.shap_explainer import FlowExplainer, stratified_sample
from src.features.feature_pipeline import build_preprocessor


def _pipeline() -> tuple[Pipeline, pd.DataFrame]:
    rng = np.random.default_rng(0)
    n = 400
    attack = rng.random(n) < 0.3
    X = pd.DataFrame({
        "Flow Duration": np.where(attack, rng.normal(10, 1, n), rng.normal(100, 10, n)),
        "Flow Bytes/s": rng.normal(50, 5, n),  # uninformative
        "Destination Port": rng.integers(1, 1000, n).astype(float),
    })
    prep = build_preprocessor()
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(prep.fit_transform(X), attack.astype(int))
    return Pipeline([("preprocess", prep), ("model", model)]), X


def test_explanation_is_additive_and_points_at_the_informative_feature() -> None:
    pipe, X = _pipeline()
    exp = FlowExplainer(pipe)
    attack_like = X[X["Flow Duration"] < 30].iloc[[0]]
    e = exp.explain(attack_like)
    assert e.is_attack and e.probability > 0.5
    assert e.contributions.iloc[0]["feature"] == "Flow Duration"
    assert e.contributions.iloc[0]["shap"] > 0
    assert abs(e.base_value + e.contributions["shap"].sum() - e.probability) < 1e-6
    # Excluded input (port) never appears in an explanation.
    assert "Destination Port" not in set(e.contributions["feature"])
    d = e.to_dict(k=2)
    assert len(d["top_features"]) == 2 and "not proof" in d["summary"]


def test_stratified_sample_caps_each_label() -> None:
    df = pd.DataFrame({"label": ["BENIGN"] * 50 + ["Bot"] * 3 + ["DDoS"] * 20, "x": range(73)})
    s = stratified_sample(df, per_label=5, benign=10)
    assert s["label"].value_counts().to_dict() == {"BENIGN": 10, "DDoS": 5, "Bot": 3}
