"""Loads the saved models once and applies the Phase 5 hybrid policy to flows.

Which classifier, and why
-------------------------
Phase 3 saved two classifiers: ``binary_best`` (a Decision Tree, picked by
validation F1) and ``binary_random_forest``. The API uses the Random Forest:

* Phase 5's hybrid evaluation and Phase 6's SHAP explanations were both done
  with it, so the API serves exactly the system those phases measured.
* Its test F1 is within 0.001 of the tree's (0.9959 vs 0.9965), and its
  probabilities are averages over 100 trees instead of mostly 0 or 1, which
  makes the score and the explanation more informative.
* Cost: about 10 MB on disk instead of 60 KB, and slower scoring. Both are
  fine for a demo that scores flows on request.

Input contract
--------------
A flow is a mapping from CIC-IDS2017 column name to a number. Every feature
the models were trained on must be present (the list is in ``/model/info``),
including ``Destination Port``, which the models then drop: the contract
mirrors the CSV columns so a row can be sent as-is. Names are compared after
stripping surrounding whitespace, as in the CSV headers. ``null`` is allowed
and means "missing"; the pipeline fills it with the training median, and the
response lists those features so the caller knows a value was imputed.
``Infinity`` occurs in the raw data (zero-duration flows) and is accepted;
the pipeline treats it as missing as well. ``Label`` is ignored so labelled
rows can be replayed. Any other unknown name is rejected, because it is
usually a typo for a feature that would otherwise be silently imputed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.schemas import IGNORED_KEYS
from src.anomaly.isolation_forest import anomaly_score, hybrid_decision
from src.explainability.shap_explainer import FlowExplainer

CLASSIFIER_FILE = "binary_random_forest.joblib"
ANOMALY_FILE = "isolation_forest.joblib"
CLASSIFIER_THRESHOLD = 0.5


class ModelNotAvailable(RuntimeError):
    pass


class InvalidFlow(ValueError):
    """Raised with a list of ``{"flow": index, "error": message}`` problems."""

    def __init__(self, problems: list[dict]):
        super().__init__(f"{len(problems)} invalid flow(s)")
        self.problems = problems


@dataclass
class Scores:
    attack_probability: np.ndarray
    anomaly_score: np.ndarray
    decision: np.ndarray


class Detector:
    def __init__(self, models_dir: Path):
        models_dir = Path(models_dir)
        clf_path, if_path = models_dir / CLASSIFIER_FILE, models_dir / ANOMALY_FILE
        missing = [p.name for p in (clf_path, if_path) if not p.exists()]
        if missing:
            raise ModelNotAvailable(
                f"{', '.join(missing)} not found in {models_dir.resolve()}. Train them first: "
                "python -m src.models.train (Phase 3) and python -m src.anomaly.isolation_forest (Phase 5)."
            )
        self.classifier = joblib.load(clf_path)
        bundle = joblib.load(if_path)
        self.anomaly_detector = bundle["detector"]
        self.anomaly_threshold = float(bundle["threshold"])
        self.anomaly_target_fpr = float(bundle["target_fpr"])
        self.explainer = FlowExplainer(self.classifier)
        self.files = {p.name: p.stat().st_size for p in (clf_path, if_path)}

        self.input_features = list(self.classifier.named_steps["preprocess"].named_steps["schema"].feature_names_in_)
        if_features = list(self.anomaly_detector.named_steps["preprocess"].named_steps["schema"].feature_names_in_)
        if if_features != self.input_features:
            raise ModelNotAvailable("The classifier and the anomaly detector expect different input features; "
                                    "retrain both from the same processed dataset.")
        self._known = set(self.input_features)

    # ------------------------------------------------------------------ input
    def to_frame(self, flows: list[dict[str, float | None]]) -> tuple[pd.DataFrame, list[list[str]]]:
        """Validate flows and return (model input, per-flow list of missing-value features)."""
        problems, rows, missing_values = [], [], []
        for i, flow in enumerate(flows):
            clean = {str(k).strip(): v for k, v in flow.items()}
            unknown = sorted(set(clean) - self._known - IGNORED_KEYS)
            absent = [f for f in self.input_features if f not in clean]
            if unknown:
                problems.append({"flow": i, "error": f"Unknown feature name(s): {unknown}"})
            if absent:
                problems.append({"flow": i, "error": f"Missing {len(absent)} required feature(s): {absent}"})
            if unknown or absent:
                continue
            rows.append([np.nan if clean[f] is None else float(clean[f]) for f in self.input_features])
            missing_values.append([f for f in self.input_features
                                   if clean[f] is None or math.isinf(clean[f])])
        if problems:
            raise InvalidFlow(problems)
        return pd.DataFrame(rows, columns=self.input_features, dtype="float64"), missing_values

    # ---------------------------------------------------------------- scoring
    def score(self, frame: pd.DataFrame) -> Scores:
        p = self.classifier.predict_proba(frame)[:, 1]
        a = anomaly_score(self.anomaly_detector, frame)
        return Scores(p, a, hybrid_decision(p, a, self.anomaly_threshold, CLASSIFIER_THRESHOLD))

    def explain(self, frame: pd.DataFrame, k: int = 10) -> dict:
        return self.explainer.explain(frame, threshold=CLASSIFIER_THRESHOLD).to_dict(k=k)

    # ------------------------------------------------------------------- info
    def info(self) -> dict:
        rf = self.classifier.named_steps["model"]
        iforest = self.anomaly_detector.named_steps["iforest"]
        prep = self.classifier.named_steps["preprocess"]
        return {
            "classifier": {
                "type": type(rf).__name__,
                "n_estimators": rf.n_estimators,
                "threshold": CLASSIFIER_THRESHOLD,
                "positive_class": "attack (any non-BENIGN CIC-IDS2017 label)",
            },
            "anomaly_detector": {
                "type": type(iforest).__name__,
                "n_estimators": iforest.n_estimators,
                "trained_on": "BENIGN training flows only",
                "threshold": self.anomaly_threshold,
                "threshold_rule": f"validation BENIGN score quantile for a {self.anomaly_target_fpr:.0%} false-positive rate",
            },
            "hybrid_policy": [
                {"rule": "attack_probability >= 0.5", "decision": "known_attack"},
                {"rule": "otherwise, anomaly_score >= anomaly threshold", "decision": "anomaly"},
                {"rule": "otherwise", "decision": "benign"},
            ],
            "input_features": self.input_features,
            "model_features": list(prep.get_feature_names_out()),
            "excluded_features": list(prep.named_steps["exclude"].excluded_),
            "ignored_keys": sorted(IGNORED_KEYS),
            "files": self.files,
        }
