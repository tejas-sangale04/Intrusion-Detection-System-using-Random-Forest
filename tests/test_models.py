from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import predict, train
from src.models.evaluate import binary_metrics, per_label_rates


def test_binary_metrics_on_known_counts() -> None:
    y = np.array([0, 0, 0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.7, 0.3, 0.9, 0.4])
    m = binary_metrics(y, score)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 1, 1, 1)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["fpr"] == pytest.approx(0.25)
    assert m["accuracy"] == pytest.approx(4 / 6)


def test_majority_class_shows_why_accuracy_misleads() -> None:
    y = np.array([0] * 95 + [1] * 5)
    m = binary_metrics(y, np.zeros(100))
    assert m["accuracy"] == pytest.approx(0.95)
    assert m["recall"] == 0 and m["f1"] == 0


def test_per_label_rates() -> None:
    out = per_label_rates(pd.Series(["BENIGN", "BENIGN", "DDoS", "DDoS"]), np.array([1, 0, 1, 1]))
    assert out.loc["BENIGN", "flagged_rate"] == 0.5
    assert out.loc["DDoS", "flagged_rate"] == 1.0


@pytest.fixture
def processed(tmp_path: Path) -> Path:
    """A small synthetic processed dataset in the Phase 2 format (not real data)."""
    rng = np.random.default_rng(0)
    n = 3000
    attack = rng.random(n) < 0.2
    df = pd.DataFrame(
        {
            "Destination Port": rng.integers(1, 65535, n),
            "Flow Duration": np.where(attack, rng.normal(10, 2, n), rng.normal(50, 10, n)),
            "Flow Bytes/s": np.where(attack, rng.normal(1e5, 1e4, n), rng.normal(1e3, 1e2, n)),
            "Bwd PSH Flags": 0.0,
        }
    ).astype("float32")
    df["label"] = np.where(attack, "DDoS", "BENIGN")
    df["family"] = df["label"]
    df["is_attack"] = attack.astype("int8")
    df["source_file"] = "synthetic.csv"
    df["split"] = rng.choice(["train", "val", "test"], n, p=[0.7, 0.15, 0.15])
    path = tmp_path / "flows.parquet"
    df.to_parquet(path, index=False)
    return path


def test_training_end_to_end_saves_a_usable_model(processed: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    train.main([
        "--data", str(processed), "--models-dir", str(out), "--metrics-dir", str(out),
        "--figures-dir", str(out), "--rf-trees", "10", "--latency-rows", "100",
    ])
    results = json.loads((out / "results.json").read_text())
    assert results["best"] != "majority_class"
    assert results["results"][results["best"]]["test"]["f1"] > 0.95
    assert (out / "confusion_matrices.png").stat().st_size > 0

    pipe, card = predict.load_model(out)
    raw = pd.read_parquet(processed).query("split == 'test'")
    # Whitespace in headers (as in the raw CSVs) and extra columns are tolerated.
    raw = raw.rename(columns={"Flow Duration": " Flow Duration"})
    preds = predict.predict_frame(pipe, raw, card["threshold"])
    assert ((preds["is_attack"] == raw["is_attack"].to_numpy()).mean()) > 0.95
