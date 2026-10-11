"""Phase 5: Isolation Forest anomaly detector and the hybrid decision policy.

Run after Phase 2 (and Phase 4, for the hybrid comparison on unseen families):

    python -m src.anomaly.isolation_forest

Anomaly detector
----------------
* **Trained on BENIGN training flows only.** It never sees an attack label,
  so its behaviour on any attack family is the same whether or not that
  family was "held out": it is the natural comparison for Phase 4.
* **Features:** the same schema as the classifier (``Destination Port``
  excluded, constant/duplicate columns dropped, inf -> NaN -> median), then
  the scaled variant's signed log transform and standardisation. Isolation
  Forest picks split points uniformly between a feature's min and max, so on
  raw heavy-tailed rates almost every split would fall in an empty tail; the
  log transform spreads the data so splits are informative.
* **Score:** ``-score_samples``; higher means easier to isolate, i.e. more
  unusual relative to benign training traffic.
* **Threshold:** chosen on **validation BENIGN flows only**, as the score
  quantile that gives a target false-positive rate (default 1%). No attack
  label is used to set it, so the threshold is honest for unseen attacks.
  The cost is that recall is whatever falls out; it is not optimised.
* An anomaly is "unusual compared to the benign traffic seen in training",
  not proof of malice. New benign applications, backups or misconfigured
  hosts are anomalies too.

Hybrid policy (transparent, rule based)
---------------------------------------
For every flow, with ``p`` the classifier's attack probability and ``a`` the
anomaly score:

=====================  ===================  =========================================
classifier             anomaly detector     decision
=====================  ===================  =========================================
p >= 0.5               any                  ``known_attack``: alert, high confidence
p < 0.5                a >= threshold       ``anomaly``: send to analyst review
p < 0.5                a < threshold        ``benign``
=====================  ===================  =========================================

The classifier wins when it is confident because it is far more precise on
attack types it was trained on. The anomaly detector only adds alerts the
classifier would have missed, and they are labelled differently so an
analyst knows which ones are pattern matches and which are "unusual, please
look". Disagreements (classifier says benign, detector says anomalous) are
exactly the ``anomaly`` bucket; the opposite disagreement stays
``known_attack``.

Evaluation reports each detector alone and the hybrid on (a) the random
test split and (b) each Phase 4 held-out family, where the classifier was
trained without that family. The question for (b) is how many unseen-family
flows the hybrid catches that the classifier alone misses, and at what cost
in extra alerts on BENIGN traffic.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.data.preprocess import feature_columns, load_split
from src.features.feature_pipeline import build_preprocessor
from src.models.evaluate import per_label_rates

logger = logging.getLogger(__name__)

SEED = 42
DECISIONS = ("benign", "anomaly", "known_attack")


def build_detector(n_estimators: int = 200, max_samples: int | str = 256) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(scale=True)),
        ("iforest", IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                                    random_state=SEED, n_jobs=-1)),
    ])


def anomaly_score(detector: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Higher = more anomalous."""
    return -detector.score_samples(X)


def threshold_for_fpr(benign_scores: np.ndarray, target_fpr: float) -> float:
    """Score above which ``target_fpr`` of the given benign flows fall."""
    return float(np.quantile(benign_scores, 1 - target_fpr))


def hybrid_decision(p_attack: np.ndarray, a_score: np.ndarray, a_threshold: float,
                    p_threshold: float = 0.5) -> np.ndarray:
    decision = np.full(len(p_attack), "benign", dtype=object)
    decision[a_score >= a_threshold] = "anomaly"
    decision[p_attack >= p_threshold] = "known_attack"
    return decision


def detection_summary(is_attack: np.ndarray, flagged: np.ndarray, score: np.ndarray | None = None) -> dict:
    is_attack = np.asarray(is_attack).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    tp, fp = int((flagged & is_attack).sum()), int((flagged & ~is_attack).sum())
    fn, tn = int((~flagged & is_attack).sum()), int((~flagged & ~is_attack).sum())
    out = {
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "alerts": tp + fp, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    out["f1"] = (2 * out["precision"] * out["recall"] / (out["precision"] + out["recall"])
                 if tp else 0.0)
    if score is not None and is_attack.any() and (~is_attack).any():
        out["roc_auc"] = float(roc_auc_score(is_attack, score))
        out["pr_auc"] = float(average_precision_score(is_attack, score))
    return out


def evaluate_hybrid_on_lofo(lofo: pd.DataFrame, a_score: np.ndarray, a_threshold: float) -> pd.DataFrame:
    """Per held-out family: classifier alone, detector alone, hybrid.

    ``lofo`` is Phase 4's test-score table (same row order as the test split)
    with one column per ``model|family`` experiment.
    """
    rows = []
    benign = lofo["is_attack"].to_numpy() == 0
    a_flag = a_score >= a_threshold
    for col in [c for c in lofo.columns if "|" in c]:
        model, fam = col.split("|", 1)
        held = (lofo["family"] == fam).to_numpy()
        p = lofo[col].to_numpy()
        c_flag = p >= 0.5
        h = hybrid_decision(p, a_score, a_threshold)
        h_flag = h != "benign"
        rows.append({
            "model": model, "held_out": fam, "held_out_rows": int(held.sum()),
            "classifier_recall": c_flag[held].mean(),
            "detector_recall": a_flag[held].mean(),
            "hybrid_recall": h_flag[held].mean(),
            "hybrid_recall_as_anomaly": (h[held] == "anomaly").mean(),
            "classifier_fpr": c_flag[benign].mean(),
            "hybrid_fpr": h_flag[benign].mean(),
        })
    return pd.DataFrame(rows)


def plot_scores(score: np.ndarray, labels: pd.Series, threshold: float, out: Path) -> None:
    order = labels.value_counts().index
    fig, ax = plt.subplots(figsize=(9, 0.35 * len(order) + 1.5))
    data = [score[(labels == l).to_numpy()] for l in order]
    ax.boxplot(data, vert=False, showfliers=False, tick_labels=list(order))
    ax.axvline(threshold, color="#E45756", linestyle="--", label="threshold (1% validation FPR)")
    ax.set_xlabel("anomaly score (higher = more unusual)")
    ax.set_title("Isolation Forest scores on the test split, by label")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Isolation Forest detector and hybrid policy evaluation.")
    parser.add_argument("--data", default="data/processed/flows.parquet")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--metrics-dir", default="reports/metrics/anomaly")
    parser.add_argument("--figures-dir", default="reports/figures/anomaly")
    parser.add_argument("--lofo-scores", default="data/processed/lofo_test_scores.parquet")
    parser.add_argument("--classifier", default="models/binary_random_forest.joblib")
    parser.add_argument("--target-fpr", type=float, nargs="*", default=[0.01, 0.05])
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-samples", type=int, default=256)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    metrics_dir, figures_dir, models_dir = Path(args.metrics_dir), Path(args.figures_dir), Path(args.models_dir)
    for d in (metrics_dir, figures_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    train = load_split(args.data, "train")
    feats = feature_columns(train)
    benign_train = train.loc[train["is_attack"] == 0, feats]
    del train
    start = time.perf_counter()
    detector = build_detector(args.n_estimators, args.max_samples).fit(benign_train)
    fit_seconds = time.perf_counter() - start
    benign_train_rows = len(benign_train)
    logger.info("Isolation Forest fitted on %d BENIGN training flows in %.1fs", benign_train_rows, fit_seconds)
    del benign_train

    val = load_split(args.data, "val")
    val_benign_scores = anomaly_score(detector, val.loc[val["is_attack"] == 0, feats])
    thresholds = {f: threshold_for_fpr(val_benign_scores, f) for f in args.target_fpr}
    del val

    test = load_split(args.data, "test")
    a_test = anomaly_score(detector, test[feats])
    y = test["is_attack"].to_numpy()
    results = {"fit_seconds": round(fit_seconds, 1), "benign_train_rows": benign_train_rows,
               "thresholds": thresholds, "detector": {}, "hybrid_random_split": {}}
    per_label = {}
    for f, t in thresholds.items():
        results["detector"][str(f)] = detection_summary(y, a_test >= t, a_test)
        per_label[f"detector@{f:.0%}"] = per_label_rates(test["label"], a_test >= t)["flagged_rate"]
        logger.info("detector @%.0f%% val FPR: test recall %.4f precision %.4f FPR %.4f",
                    100 * f, results["detector"][str(f)]["recall"], results["detector"][str(f)]["precision"],
                    results["detector"][str(f)]["fpr"])

    primary = min(thresholds)  # 1% by default
    joblib.dump({"detector": detector, "threshold": thresholds[primary], "target_fpr": primary},
                models_dir / "isolation_forest.joblib", compress=3)
    plot_scores(a_test, test["label"], thresholds[primary], figures_dir / "anomaly_scores_by_label.png")

    if Path(args.classifier).exists():
        clf = joblib.load(args.classifier)
        p_test = clf.predict_proba(test[feats])[:, 1]
        for f, t in thresholds.items():
            h = hybrid_decision(p_test, a_test, t)
            results["hybrid_random_split"][str(f)] = {
                "classifier_only": detection_summary(y, p_test >= 0.5),
                "hybrid": detection_summary(y, h != "benign"),
                "decision_counts": {d: int((h == d).sum()) for d in DECISIONS},
            }
        per_label["classifier"] = per_label_rates(test["label"], p_test >= 0.5)["flagged_rate"]
        per_label[f"hybrid@{primary:.0%}"] = per_label_rates(
            test["label"], hybrid_decision(p_test, a_test, thresholds[primary]) != "benign")["flagged_rate"]
    else:
        logger.warning("%s not found; skipping hybrid on the random split", args.classifier)

    pd.DataFrame(per_label).assign(rows=test["label"].value_counts()).sort_values("rows", ascending=False).to_csv(
        metrics_dir / "per_label_test.csv")

    if Path(args.lofo_scores).exists():
        lofo = pd.read_parquet(args.lofo_scores)
        if not (lofo["label"].to_numpy() == test["label"].to_numpy()).all():
            raise ValueError("Phase 4 scores are not aligned with the current test split; rerun Phase 4")
        table = pd.concat(
            [evaluate_hybrid_on_lofo(lofo, a_test, t).assign(target_fpr=f) for f, t in thresholds.items()],
            ignore_index=True,
        )
        table.to_csv(metrics_dir / "hybrid_held_out_families.csv", index=False)
        results["hybrid_held_out_families"] = table.to_dict(orient="records")
    else:
        logger.warning("%s not found; run Phase 4 first for the held-out-family comparison", args.lofo_scores)

    (metrics_dir / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
