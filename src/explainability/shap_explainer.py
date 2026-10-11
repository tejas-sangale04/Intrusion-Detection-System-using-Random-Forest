"""Phase 6: SHAP explanations for the Random Forest detector.

Run after Phase 3:

    python -m src.explainability.shap_explainer

How to read SHAP values here
----------------------------
For one flow, each feature gets a number in units of predicted attack
probability. The model's output for that flow equals the average output over
the background data (the "base value") plus the sum of all features' SHAP
values. A positive value means "this feature's value pushed the prediction
towards attack", negative means "towards benign".

What SHAP does **not** say: that the feature caused the attack, or that
changing it would change the traffic's nature. It explains the *model*, which
learned correlations in CIC-IDS2017 (including lab artefacts). Correlated
features also share credit in ways that can look arbitrary: if two columns
carry the same information, either may receive it.

Cost and the sampling trade-off
-------------------------------
TreeExplainer is exact for tree ensembles, but its cost grows with tree size.
The saved forest has ~5,900 nodes and depth ~41 per tree, giving about 0.3 s
per explained flow on 4 cores. That is fine for explaining one alert on
demand (the API case), but explaining the whole test split (378k flows) would
take over a day. Global importance is therefore computed on a stratified
sample: up to ``--per-label`` flows of every label plus ``--benign`` BENIGN
flows. Stratifying keeps rare attacks represented; the cost is that global
averages are over that sample's class mix, not the real one, and are
reported per class as well as overall for that reason.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.data.preprocess import feature_columns, load_split

logger = logging.getLogger(__name__)

SEED = 42


@dataclass
class FlowExplanation:
    probability: float
    is_attack: bool
    base_value: float
    contributions: pd.DataFrame  # feature, value, shap; sorted by |shap|

    def top(self, k: int = 5) -> pd.DataFrame:
        return self.contributions.head(k)

    def text(self, k: int = 3) -> str:
        verdict = "flagged as attack" if self.is_attack else "classified as benign"
        towards_attack = self.contributions[self.contributions["shap"] > 0].head(k)
        towards_benign = self.contributions[self.contributions["shap"] < 0].head(k)

        def fmt(rows: pd.DataFrame) -> str:
            return "; ".join(f"{r.feature} = {r.value:,.4g} ({r.shap:+.3f})" for r in rows.itertuples()) or "none"

        return (
            f"The model {verdict} (attack probability {self.probability:.3f}, threshold 0.5; "
            f"average over reference traffic {self.base_value:.3f}). "
            f"Largest pushes towards attack: {fmt(towards_attack)}. "
            f"Largest pushes towards benign: {fmt(towards_benign)}. "
            "These describe what the model relied on, not proof of what the traffic is."
        )

    def to_dict(self, k: int = 10) -> dict:
        return {
            "probability": self.probability,
            "is_attack": self.is_attack,
            "base_value": self.base_value,
            "top_features": self.top(k).to_dict(orient="records"),
            "summary": self.text(),
        }


class FlowExplainer:
    """Wraps a fitted ``Pipeline(preprocess, model)`` with a SHAP TreeExplainer."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.preprocess = pipeline.named_steps["preprocess"]
        self.model = pipeline.named_steps["model"]
        # tree_path_dependent: uses the trees' own training-sample counts as the
        # reference distribution, so no separate background dataset is needed.
        self.explainer = shap.TreeExplainer(self.model)
        self.feature_names = list(self.preprocess.get_feature_names_out())

    def _attack_column(self, values) -> np.ndarray:
        arr = np.asarray(values)
        return arr[..., 1] if arr.ndim == 3 else arr

    def shap_values(self, raw: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Return (model-space features, SHAP values for the attack class, base value)."""
        X = self.preprocess.transform(raw).astype(np.float32)
        values = self._attack_column(self.explainer.shap_values(X, check_additivity=False))
        base = np.atleast_1d(self.explainer.expected_value)
        base_attack = float(base[1] if len(base) > 1 else base[0])
        return X, values, np.full(len(X), base_attack)

    def explain(self, raw_row: pd.DataFrame, threshold: float = 0.5) -> FlowExplanation:
        if len(raw_row) != 1:
            raise ValueError("explain() expects exactly one flow")
        X, values, base = self.shap_values(raw_row)
        prob = float(self.model.predict_proba(X)[0, 1])
        contrib = pd.DataFrame({"feature": self.feature_names, "value": X.iloc[0].to_numpy(), "shap": values[0]})
        contrib = contrib.reindex(contrib["shap"].abs().sort_values(ascending=False).index).reset_index(drop=True)
        return FlowExplanation(prob, prob >= threshold, float(base[0]), contrib)


def stratified_sample(df: pd.DataFrame, per_label: int, benign: int, seed: int = SEED) -> pd.DataFrame:
    parts = []
    for label, group in df.groupby("label"):
        n = benign if label == "BENIGN" else per_label
        parts.append(group.sample(n=min(n, len(group)), random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed)


def plot_global(importance: pd.Series, out: Path, k: int = 15) -> None:
    top = importance.head(k)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * k + 1.2))
    ax.barh(top.index, top.to_numpy(), color="#4C78A8")
    ax.set_xlabel("mean |SHAP value| (attack probability)")
    ax.set_title("Global feature importance, Random Forest (stratified sample)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_by_label(table: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(0.5 * table.shape[1] + 4, 0.4 * table.shape[0] + 2))
    im = ax.imshow(table.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(table.shape[0]), table.index, fontsize=8)
    fig.colorbar(im, ax=ax, label="mean |SHAP|")
    ax.set_title("Which features the model relies on, per true label")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SHAP explanations for the Random Forest detector.")
    parser.add_argument("--data", default="data/processed/flows.parquet")
    parser.add_argument("--model", default="models/binary_random_forest.joblib")
    parser.add_argument("--metrics-dir", default="reports/metrics/explainability")
    parser.add_argument("--figures-dir", default="reports/figures/explainability")
    parser.add_argument("--per-label", type=int, default=40)
    parser.add_argument("--benign", type=int, default=200)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    metrics_dir, figures_dir = Path(args.metrics_dir), Path(args.figures_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    explainer = FlowExplainer(joblib.load(args.model))
    test = load_split(args.data, "test")
    feats = feature_columns(test)
    sample = stratified_sample(test, args.per_label, args.benign)
    logger.info("explaining %d test flows (%s)", len(sample), sample["label"].value_counts().to_dict())

    start = time.perf_counter()
    X, values, base = explainer.shap_values(sample[feats])
    seconds = time.perf_counter() - start
    logger.info("SHAP for %d flows in %.1fs (%.3fs per flow)", len(sample), seconds, seconds / len(sample))

    # Additivity check: base + sum(SHAP) must reproduce the model output.
    prob = explainer.model.predict_proba(X)[:, 1]
    max_err = float(np.abs(base + values.sum(axis=1) - prob).max())
    logger.info("max |base + sum(SHAP) - probability| = %.2e", max_err)

    sv = pd.DataFrame(values, columns=explainer.feature_names, index=sample.index)
    importance = sv.abs().mean().sort_values(ascending=False)
    importance.rename("mean_abs_shap").to_csv(metrics_dir / "global_importance.csv")
    by_label = sv.abs().groupby(sample["label"]).mean()[importance.index[:20]]
    by_label.to_csv(metrics_dir / "importance_by_label.csv")
    plot_global(importance, figures_dir / "global_importance.png")
    plot_by_label(by_label, figures_dir / "importance_by_label.png")

    plt.figure()
    shap.summary_plot(values, X, max_display=15, show=False)
    plt.title("SHAP values per flow (stratified sample)")
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_summary.png", dpi=150)
    plt.close("all")

    # A few worked single-flow explanations for the README / dashboard demo.
    examples = {}
    for label in ("DDoS", "PortScan", "Bot", "BENIGN"):
        rows = sample[sample["label"] == label]
        if len(rows):
            examples[label] = explainer.explain(rows[feats].iloc[[0]]).to_dict(k=5)
    (metrics_dir / "examples.json").write_text(json.dumps(examples, indent=2, default=float), encoding="utf-8")
    (metrics_dir / "summary.json").write_text(json.dumps({
        "flows_explained": int(len(sample)),
        "label_counts": sample["label"].value_counts().to_dict(),
        "seconds": round(seconds, 1),
        "seconds_per_flow": round(seconds / len(sample), 4),
        "max_additivity_error": max_err,
        "base_value": float(base[0]),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
