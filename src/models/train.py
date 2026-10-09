"""Phase 3: train and compare baseline binary classifiers (BENIGN = 0, attack = 1).

Run after Phase 2:

    python -m src.models.train

Protocol
--------
* Preprocessing pipelines are fitted on the **training split only**
  (``src.features.feature_pipeline``), then applied unchanged to validation
  and test.
* Every model is fitted on the training split. The best model is chosen by
  **validation F1**; the test split is used once, for the final numbers, and
  never for choosing anything.
* A majority-class dummy model is included as a reference: it shows how much
  of the accuracy figure is free.
* The threshold is the default 0.5 for every model. Tuning it belongs on the
  validation split and is done in later phases together with the anomaly
  detector.

Limits of this experiment: the split is random over all days, so train and
test share the same attack campaigns, hosts and recording conditions. These
numbers measure in-distribution detection only. Phase 4 tests generalisation
to unseen days and attack families.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data.preprocess import feature_columns, load_split
from src.features.feature_pipeline import build_preprocessor
from src.models.evaluate import binary_metrics, per_label_rates

logger = logging.getLogger(__name__)

SEED = 42


def model_specs(rf_trees: int = 100) -> dict[str, tuple[object, bool]]:
    """name -> (unfitted estimator, needs scaled features)."""
    return {
        "majority_class": (DummyClassifier(strategy="most_frequent"), False),
        "logistic_regression": (LogisticRegression(max_iter=2000, random_state=SEED), True),
        "decision_tree": (DecisionTreeClassifier(random_state=SEED), False),
        "random_forest": (RandomForestClassifier(n_estimators=rf_trees, n_jobs=-1, random_state=SEED), False),
        "hist_gradient_boosting": (HistGradientBoostingClassifier(random_state=SEED), False),
    }


def _timed(fn, *args):
    start = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - start


def plot_confusion_matrices(results: dict[str, dict], out: Path) -> None:
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.2))
    for ax, name in zip(np.atleast_1d(axes), names, strict=True):
        m = results[name]["test"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["BENIGN", "attack"], fontsize=8)
        ax.set_yticks([0, 1], ["BENIGN", "attack"], fontsize=8)
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_ylabel("actual", fontsize=8)
        ax.set_title(name.replace("_", " "), fontsize=9)
    fig.suptitle("Confusion matrices on the test split (threshold 0.5)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and compare baseline binary classifiers.")
    parser.add_argument("--data", default="data/processed/flows.parquet")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--metrics-dir", default="reports/metrics/baselines")
    parser.add_argument("--figures-dir", default="reports/figures/baselines")
    parser.add_argument("--models", nargs="*", help="subset of models to train (default: all)")
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--train-sample", type=int, default=0, help="subsample training rows (0 = all), for quick runs")
    parser.add_argument("--latency-rows", type=int, default=100_000)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    metrics_dir, figures_dir, models_dir = Path(args.metrics_dir), Path(args.figures_dir), Path(args.models_dir)
    for d in (metrics_dir, figures_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    splits = {s: load_split(args.data, s) for s in ("train", "val", "test")}
    if args.train_sample:
        splits["train"] = splits["train"].sample(n=args.train_sample, random_state=SEED)
    feats = feature_columns(splits["train"])
    y = {s: df["is_attack"].to_numpy() for s, df in splits.items()}
    logger.info("rows: %s; %d input features", {s: len(df) for s, df in splits.items()}, len(feats))

    prep = {}
    X = {}
    for scaled in (False, True):
        p, secs = _timed(build_preprocessor(scale=scaled).fit, splits["train"][feats])
        prep[scaled] = (p, secs)
        X[scaled] = {s: p.transform(df[feats]).astype(np.float32) for s, df in splits.items()}
        logger.info("preprocessor scale=%s fitted in %.1fs -> %d features", scaled, secs, X[scaled]["train"].shape[1])

    specs = model_specs(args.rf_trees)
    names = args.models or list(specs)
    results: dict[str, dict] = {}
    pipelines: dict[str, Pipeline] = {}
    per_label = {}
    latency_frame = splits["test"][feats].iloc[: args.latency_rows]

    for name in names:
        model, scaled = specs[name]
        logger.info("training %s ...", name)
        _, fit_secs = _timed(model.fit, X[scaled]["train"], y["train"])
        pipe = Pipeline([("preprocess", prep[scaled][0]), ("model", model)])
        pipelines[name] = pipe

        res = {"fit_seconds": round(fit_secs, 2), "scaled_features": scaled}
        for split in ("val", "test"):
            score = model.predict_proba(X[scaled][split])[:, 1]
            res[split] = binary_metrics(y[split], score)
            if split == "test":
                per_label[name] = per_label_rates(splits["test"]["label"], score >= 0.5)
        # End-to-end latency: raw feature rows in, attack probability out.
        _, infer_secs = _timed(pipe.predict_proba, latency_frame)
        res["inference_us_per_flow"] = round(1e6 * infer_secs / len(latency_frame), 2)
        results[name] = res
        t = res["test"]
        logger.info(
            "%s: fit %.1fs | val F1 %.4f | test F1 %.4f precision %.4f recall %.4f FPR %.5f PR-AUC %.4f",
            name, fit_secs, res["val"]["f1"], t["f1"], t["precision"], t["recall"], t["fpr"], t["pr_auc"],
        )

    candidates = [n for n in results if n != "majority_class"]
    best = max(candidates, key=lambda n: results[n]["val"]["f1"])
    logger.info("best by validation F1: %s", best)

    rows = []
    for name, res in results.items():
        for split in ("val", "test"):
            rows.append({"model": name, "split": split, **res[split], "fit_seconds": res["fit_seconds"],
                         "inference_us_per_flow": res["inference_us_per_flow"]})
    pd.DataFrame(rows).to_csv(metrics_dir / "comparison.csv", index=False)
    pd.concat({n: d["flagged_rate"] for n, d in per_label.items()}, axis=1).assign(
        rows=next(iter(per_label.values()))["rows"]
    ).to_csv(metrics_dir / "per_label_test.csv")
    plot_confusion_matrices(results, figures_dir / "confusion_matrices.png")

    joblib.dump(pipelines[best], models_dir / "binary_best.joblib", compress=3)
    # The project's hybrid detector (Phase 5) is specified around a Random
    # Forest, so it is kept even when another model scores higher here.
    if "random_forest" in pipelines and best != "random_forest":
        joblib.dump(pipelines["random_forest"], models_dir / "binary_random_forest.joblib", compress=3)
    card = {
        "model": best,
        "selected_by": "validation F1 at threshold 0.5",
        "threshold": 0.5,
        "positive_class": "attack (any non-BENIGN label)",
        "input_features": feats,
        "model_features": list(pipelines[best].named_steps["preprocess"].get_feature_names_out()),
        "train_rows": int(len(splits["train"])),
        "metrics": results[best],
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "data": str(args.data),
        "seed": SEED,
    }
    (models_dir / "binary_best.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    (metrics_dir / "results.json").write_text(json.dumps({"best": best, "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
