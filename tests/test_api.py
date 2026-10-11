"""API tests on tiny models trained on synthetic flows (no dataset needed).

The numbers here are invented to exercise the code; they say nothing about
detection quality on CIC-IDS2017.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from backend.config import Settings
from backend.main import create_app
from src.anomaly.isolation_forest import anomaly_score, build_detector, threshold_for_fpr
from src.features.feature_pipeline import build_preprocessor

FEATURES = ["Destination Port", "Flow Duration", "Flow Bytes/s", "Total Fwd Packets"]
BENIGN = {"Destination Port": 443, "Flow Duration": 100.0, "Flow Bytes/s": 50.0, "Total Fwd Packets": 10}
ATTACK = {"Destination Port": 80, "Flow Duration": 10.0, "Flow Bytes/s": 50.0, "Total Fwd Packets": 10}
UNUSUAL = {"Destination Port": 443, "Flow Duration": 100.0, "Flow Bytes/s": 1e6, "Total Fwd Packets": 1e6}


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory) -> Path:
    rng = np.random.default_rng(0)
    n = 600
    attack = rng.random(n) < 0.3
    X = pd.DataFrame({
        "Destination Port": rng.integers(1, 1000, n).astype(float),
        "Flow Duration": np.where(attack, rng.normal(10, 1, n), rng.normal(100, 10, n)),
        "Flow Bytes/s": rng.normal(50, 5, n),
        "Total Fwd Packets": rng.normal(10, 1, n),
    })
    prep = build_preprocessor()
    rf = RandomForestClassifier(n_estimators=20, random_state=0).fit(prep.fit_transform(X), attack.astype(int))
    det = build_detector(n_estimators=50).fit(X[~attack])
    out = tmp_path_factory.mktemp("models")
    joblib.dump(Pipeline([("preprocess", prep), ("model", rf)]), out / "binary_random_forest.joblib")
    threshold = threshold_for_fpr(anomaly_score(det, X[~attack]), 0.01)
    joblib.dump({"detector": det, "threshold": threshold, "target_fpr": 0.01}, out / "isolation_forest.joblib")
    return out


@pytest.fixture
def client(models_dir: Path, tmp_path: Path):
    reports = tmp_path / "reports"
    (reports / "anomaly").mkdir(parents=True)
    (reports / "anomaly" / "results.json").write_text(json.dumps({"thresholds": {"0.01": 0.6}}))
    settings = Settings(models_dir=models_dir, reports_dir=reports, alert_log=tmp_path / "alerts.jsonl",
                        max_batch=5, store_size=3, cors_origins=[])
    with TestClient(create_app(settings)) as c:
        yield c


def test_health_and_model_info(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/model/info").json()
    assert info["input_features"] == FEATURES
    assert info["excluded_features"] == ["Destination Port"]
    assert "Destination Port" not in info["model_features"]
    assert info["mode"] == "offline"


def test_predict_applies_hybrid_policy(client: TestClient) -> None:
    benign = client.post("/predict", json={"features": BENIGN}).json()
    attack = client.post("/predict", json={"features": ATTACK}).json()
    unusual = client.post("/predict", json={"features": UNUSUAL}).json()
    assert benign["decision"] == "benign" and benign["attack_probability"] < 0.5
    assert attack["decision"] == "known_attack" and attack["attack_probability"] >= 0.5
    assert unusual["attack_probability"] < 0.5 and unusual["anomaly_score"] >= unusual["anomaly_threshold"]
    assert unusual["decision"] == "anomaly"
    assert attack["explanation_url"] == f"/explanations/{attack['prediction_id']}"


def test_whitespace_label_null_and_infinity_are_handled(client: TestClient) -> None:
    flow = {f" {k} ": v for k, v in BENIGN.items()} | {"Label": "BENIGN"}
    flow[" Flow Bytes/s "] = None
    r = client.post("/predict", json={"features": flow})
    assert r.status_code == 200 and r.json()["missing_values"] == ["Flow Bytes/s"]
    # Infinity is not standard JSON, but occurs in CIC-IDS2017 and Python's json emits it.
    body = json.dumps({"features": BENIGN | {"Flow Bytes/s": float("inf")}})
    r = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 200 and r.json()["missing_values"] == ["Flow Bytes/s"]


@pytest.mark.parametrize("features, fragment", [
    ({k: v for k, v in BENIGN.items() if k != "Flow Duration"}, "Missing 1 required feature(s): ['Flow Duration']"),
    (BENIGN | {"Flow Durration": 1}, "Unknown feature name(s): ['Flow Durration']"),
])
def test_wrong_feature_names_are_rejected(client: TestClient, features: dict, fragment: str) -> None:
    r = client.post("/predict", json={"features": features})
    assert r.status_code == 422
    assert fragment in r.json()["errors"][0]["error"]


@pytest.mark.parametrize("value", ["100", True, [1], {"a": 1}])
def test_non_numeric_values_are_rejected(client: TestClient, value) -> None:
    r = client.post("/predict", json={"features": BENIGN | {"Flow Duration": value}})
    assert r.status_code == 422
    assert "100" not in json.dumps(r.json())  # submitted values are not echoed back


def test_malformed_body_is_rejected(client: TestClient) -> None:
    assert client.post("/predict", content="not json", headers={"Content-Type": "application/json"}).status_code == 422
    assert client.post("/predict", json={"features": {}}).status_code == 422
    assert client.post("/predict", json={"features": BENIGN, "extra": 1}).status_code == 422


def test_batch_scores_validates_and_limits(client: TestClient) -> None:
    r = client.post("/predict/batch", json={"flows": [BENIGN, ATTACK]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2 and body["decision_counts"]["known_attack"] == 1
    bad = client.post("/predict/batch", json={"flows": [BENIGN, {"Flow Duration": 1}]})
    assert bad.status_code == 422 and bad.json()["errors"][0]["flow"] == 1
    assert client.post("/predict/batch", json={"flows": [BENIGN] * 6}).status_code == 413


def test_alerts_are_newest_first_and_exclude_benign(client: TestClient, tmp_path: Path) -> None:
    client.post("/predict", json={"features": ATTACK})
    client.post("/predict", json={"features": BENIGN})
    last = client.post("/predict", json={"features": UNUSUAL}).json()
    alerts = client.get("/alerts").json()["alerts"]
    assert [a["decision"] for a in alerts] == ["anomaly", "known_attack"]
    assert alerts[0]["prediction_id"] == last["prediction_id"]
    assert client.get("/alerts", params={"decision": "known_attack"}).json()["count"] == 1
    assert client.get("/alerts", params={"decision": "benign"}).status_code == 422
    logged = (tmp_path / "alerts.jsonl").read_text().splitlines()
    assert len(logged) == 2 and "features" not in json.loads(logged[0])


def test_store_drops_oldest_beyond_its_size(client: TestClient) -> None:
    first = client.post("/predict", json={"features": ATTACK}).json()["prediction_id"]
    for _ in range(3):
        client.post("/predict", json={"features": BENIGN})
    assert client.get(f"/explanations/{first}").status_code == 404
    assert client.get("/metrics").json()["runtime"]["predictions_evicted"] == 1


def test_explanation_matches_prediction(client: TestClient) -> None:
    pred = client.post("/predict", json={"features": ATTACK}).json()
    r = client.get(pred["explanation_url"], params={"k": 2})
    assert r.status_code == 200
    e = r.json()
    assert e["attack_probability"] == pytest.approx(pred["attack_probability"])
    assert len(e["top_features"]) == 2 and e["top_features"][0]["feature"] == "Flow Duration"
    assert "not proof" in e["summary"]
    assert client.get("/explanations/does-not-exist").status_code == 404


def test_metrics_report_files_and_runtime_counts(client: TestClient) -> None:
    client.post("/predict/batch", json={"flows": [BENIGN, ATTACK, UNUSUAL]})
    m = client.get("/metrics").json()
    off = m["offline_evaluation"]
    assert off["anomaly_and_hybrid"]["data"] == {"thresholds": {"0.01": 0.6}}
    assert off["classifier_baselines"] is None  # missing file is reported, not filled in
    assert any(p.endswith("baselines/results.json") for p in off["missing_files"])
    rt = m["runtime"]
    assert rt["flows_scored"] == 3 and rt["decision_counts"] == {"benign": 1, "anomaly": 1, "known_attack": 1}


def test_missing_models_stop_startup(tmp_path: Path) -> None:
    app = create_app(Settings(models_dir=tmp_path, reports_dir=tmp_path, cors_origins=[]))
    with pytest.raises(RuntimeError, match="binary_random_forest.joblib"):
        with TestClient(app):
            pass
