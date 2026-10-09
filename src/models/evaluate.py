"""Metrics for binary intrusion detection (attack = positive class = 1).

Why not accuracy alone: after de-duplication about 83% of flows are BENIGN,
so a model that never raises an alert is already ~83% accurate while
detecting nothing. The metrics that matter operationally are:

* **recall** (detection rate): share of attack flows that are flagged
* **precision**: share of alerts that are real attacks
* **false-positive rate**: share of BENIGN flows wrongly flagged. Even 1% FPR
  on a network with millions of benign flows a day means tens of thousands of
  false alerts, which is why it is reported separately from precision
* **F1**: harmonic mean of precision and recall at the chosen threshold
* **ROC-AUC / PR-AUC**: threshold-free ranking quality. PR-AUC (average
  precision) is the more informative one under class imbalance

Per-label detection rates are reported too, because a high overall recall can
hide a class (for example a rare web attack) that is almost never detected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both = len(np.unique(y_true)) == 2
    return {
        "threshold": threshold,
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
        "roc_auc": roc_auc_score(y_true, y_score) if has_both else float("nan"),
        "pr_auc": average_precision_score(y_true, y_score) if has_both else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def per_label_rates(labels: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """Share of each fine label predicted as attack.

    For attack labels this is the detection rate (recall); for BENIGN it is
    the false-positive rate.
    """
    frame = pd.DataFrame({"label": np.asarray(labels), "flagged": np.asarray(y_pred).astype(int)})
    out = frame.groupby("label")["flagged"].agg(rows="size", flagged="sum")
    out["flagged_rate"] = out["flagged"] / out["rows"]
    return out.sort_values("rows", ascending=False)
