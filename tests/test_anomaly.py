from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.anomaly.isolation_forest import (
    anomaly_score,
    build_detector,
    detection_summary,
    evaluate_hybrid_on_lofo,
    hybrid_decision,
    threshold_for_fpr,
)


def test_threshold_hits_target_fpr_on_benign_scores() -> None:
    scores = np.arange(1000, dtype=float)
    t = threshold_for_fpr(scores, 0.01)
    assert (scores >= t).mean() == pytest.approx(0.01, abs=0.002)


def test_hybrid_policy_table() -> None:
    p = np.array([0.9, 0.9, 0.1, 0.1])
    a = np.array([5.0, 0.0, 5.0, 0.0])
    assert hybrid_decision(p, a, a_threshold=1.0).tolist() == ["known_attack", "known_attack", "anomaly", "benign"]


def test_detection_summary_counts() -> None:
    out = detection_summary(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert (out["tp"], out["fn"], out["fp"], out["tn"]) == (1, 1, 1, 1)
    assert out["fpr"] == 0.5 and out["recall"] == 0.5


def test_detector_trained_on_benign_scores_outliers_higher() -> None:
    rng = np.random.default_rng(0)
    cols = ["Flow Duration", "Flow Bytes/s", "Total Fwd Packets"]
    benign = pd.DataFrame(rng.normal(100, 10, (2000, 3)), columns=cols)
    attack = pd.DataFrame(rng.normal(1e5, 10, (50, 3)), columns=cols)
    det = build_detector(n_estimators=50).fit(benign)
    assert anomaly_score(det, attack).min() > np.quantile(anomaly_score(det, benign), 0.99)


def test_hybrid_on_lofo_counts_unseen_family_rescued_by_detector() -> None:
    lofo = pd.DataFrame({
        "label": ["BENIGN"] * 4 + ["Bot"] * 2,
        "family": ["BENIGN"] * 4 + ["Bot"] * 2,
        "is_attack": [0, 0, 0, 0, 1, 1],
        "random_forest|Bot": [0.0, 0.0, 0.0, 0.0, 0.1, 0.1],  # classifier misses unseen Bot
    })
    a = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 0.0])  # detector catches one Bot, one false alarm
    row = evaluate_hybrid_on_lofo(lofo, a, a_threshold=1.0).iloc[0]
    assert row["classifier_recall"] == 0 and row["hybrid_recall"] == 0.5
    assert row["hybrid_fpr"] == 0.25 and row["classifier_fpr"] == 0
