"""Phase 4: does the supervised detector generalise beyond what it was trained on?

Run after Phase 2 (independent of Phase 3's saved models):

    python -m src.models.generalization

Phase 3 used a random split, so train and test share attack campaigns, hosts
and recording days. That answers "can the model recognise more of the same?".
This script asks harder questions, each with its own train/test separation:

1. **Leave one attack family out.** For each family F (DoS, DDoS, PortScan,
   Brute Force, Web Attack, Bot, Infiltration, Heartbleed), every F flow is
   removed from training. The model is trained on BENIGN plus the other
   families, then scored on the normal test split. Recall on the held-out F
   flows measures detection of an attack type the classifier never saw.
   Recall on the remaining families and FPR on BENIGN are reported alongside,
   so a drop on F is not confused with a generally worse model.
2. **Held-out day (chronological).** Train on Monday to Thursday, test on
   Friday's three files. Every Friday attack (Bot, PortScan, DDoS) is absent
   from Mon-Thu, so this is the realistic "deployed model meets next week's
   traffic" case, including any drift in benign traffic.
3. **Destination Port ablation.** Phase 2 excluded the port as a likely
   shortcut. Retraining with it on the random split shows how much the
   in-distribution score relies on it.

Each experiment fits its own preprocessing pipeline on its own training rows,
so nothing about the test rows leaks into feature selection or imputation.
The threshold stays at 0.5: choosing it on the held-out family would be
using test labels. Ranking quality on the held-out flows (ROC-AUC of held-out
attacks vs BENIGN) is reported too, since a model can rank unseen attacks
above benign traffic yet still score them below 0.5.

Isolation Forest is evaluated on the same held-out families in Phase 5, so
supervised and unsupervised detection of unseen attacks can be compared on
identical data.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.data.preprocess import feature_columns
from src.features.feature_pipeline import build_preprocessor
from src.models.evaluate import binary_metrics

logger = logging.getLogger(__name__)

SEED = 42
FRIDAY_PREFIX = "Friday"


def make_model(name: str, rf_trees: int):
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=rf_trees, n_jobs=-1, random_state=SEED)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=SEED)
    raise ValueError(name)


def fit_and_score(model_name: str, train: pd.DataFrame, test: pd.DataFrame, feats: list[str],
                  rf_trees: int, exclude: tuple[str, ...] | None = None) -> np.ndarray:
    """Fit preprocessing + model on ``train`` only; return attack probabilities for ``test``."""
    prep = build_preprocessor() if exclude is None else build_preprocessor(exclude=exclude)
    Xtr = prep.fit_transform(train[feats]).astype(np.float32)
    model = make_model(model_name, rf_trees).fit(Xtr, train["is_attack"].to_numpy())
    return model.predict_proba(prep.transform(test[feats]).astype(np.float32))[:, 1]


def summarise(test: pd.DataFrame, score: np.ndarray, held_out: pd.Series) -> dict:
    """Metrics split into held-out attacks, other (seen) attacks and BENIGN."""
    flagged = score >= 0.5
    benign = test["is_attack"].to_numpy() == 0
    held = held_out.to_numpy()
    seen = ~benign & ~held
    out = {
        "held_out_rows": int(held.sum()),
        "held_out_recall": float(flagged[held].mean()) if held.any() else float("nan"),
        "seen_attack_rows": int(seen.sum()),
        "seen_attack_recall": float(flagged[seen].mean()) if seen.any() else float("nan"),
        "benign_rows": int(benign.sum()),
        "fpr": float(flagged[benign].mean()),
        "held_out_vs_benign_roc_auc": float(
            roc_auc_score(np.r_[np.zeros(benign.sum()), np.ones(held.sum())], np.r_[score[benign], score[held]])
        ) if held.any() else float("nan"),
        "overall": binary_metrics(test["is_attack"].to_numpy(), score),
    }
    labels = test.loc[held, "label"]
    out["held_out_recall_by_label"] = (
        pd.Series(flagged[held], index=labels.index).groupby(labels).mean().round(4).to_dict()
    )
    return out


def leave_one_family_out(df: pd.DataFrame, feats: list[str], model_name: str, rf_trees: int) -> list[dict]:
    train_all, test = df[df["split"] == "train"], df[df["split"] == "test"]
    families = [f for f in df["family"].unique() if f != "BENIGN"]
    rows = []
    for fam in sorted(families):
        start = time.perf_counter()
        train = train_all[train_all["family"] != fam]
        score = fit_and_score(model_name, train, test, feats, rf_trees)
        res = summarise(test, score, test["family"] == fam)
        res.update({"experiment": "leave_one_family_out", "model": model_name, "held_out": fam,
                    "train_rows": int(len(train)), "seconds": round(time.perf_counter() - start, 1)})
        logger.info("[%s] hold out %-12s recall %.4f (seen attacks %.4f, FPR %.5f, AUC %.4f)", model_name, fam,
                    res["held_out_recall"], res["seen_attack_recall"], res["fpr"], res["held_out_vs_benign_roc_auc"])
        rows.append(res)
    return rows


def held_out_day(df: pd.DataFrame, feats: list[str], model_name: str, rf_trees: int) -> dict:
    """Train on every non-Friday row (all splits), test on every Friday row."""
    friday = df["source_file"].str.startswith(FRIDAY_PREFIX)
    train, test = df[~friday], df[friday]
    unseen_labels = set(test["label"]) - set(train["label"])
    score = fit_and_score(model_name, train, test, feats, rf_trees)
    res = summarise(test, score, test["label"].isin(unseen_labels) & (test["is_attack"] == 1))
    res.update({"experiment": "held_out_day", "model": model_name, "held_out": "Friday (all three files)",
                "unseen_labels": sorted(unseen_labels), "train_rows": int(len(train))})
    logger.info("[%s] Friday: unseen-attack recall %.4f, FPR %.5f, F1 %.4f", model_name,
                res["held_out_recall"], res["fpr"], res["overall"]["f1"])
    return res


def port_ablation(df: pd.DataFrame, feats: list[str], rf_trees: int) -> dict:
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    out = {}
    for name, exclude in (("without_port", None), ("with_port", ())):
        score = fit_and_score("random_forest", train, test, feats, rf_trees, exclude=exclude)
        m = binary_metrics(test["is_attack"].to_numpy(), score)
        flagged = pd.Series(score >= 0.5, index=test.index)
        m["recall_by_label"] = flagged[test["is_attack"] == 1].groupby(test["label"]).mean().round(4).to_dict()
        out[name] = m
        logger.info("[port ablation] %s: F1 %.4f FPR %.5f", name, m["f1"], m["fpr"])
    return out


def plot_lofo(table: pd.DataFrame, out: Path) -> None:
    models = table["model"].unique()
    fams = table["held_out"].unique()
    x = np.arange(len(fams))
    width = 0.8 / (2 * len(models))
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = {"held_out_recall": "#E45756", "seen_attack_recall": "#4C78A8"}
    for i, m in enumerate(models):
        t = table[table["model"] == m].set_index("held_out").loc[fams]
        for j, col in enumerate(("held_out_recall", "seen_attack_recall")):
            ax.bar(x + (2 * i + j - len(models)) * width + width / 2, t[col], width,
                   color=colors[col], alpha=1.0 if i == 0 else 0.55,
                   label=f"{m.replace('_', ' ')}: {'held-out family' if j == 0 else 'other attacks'}")
    ax.set_xticks(x, fams, rotation=20)
    ax.set_ylabel("recall at threshold 0.5")
    ax.set_ylim(0, 1.05)
    ax.set_title("Recall on an attack family excluded from training vs on families seen in training")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generalisation experiments for the supervised detector.")
    parser.add_argument("--data", default="data/processed/flows.parquet")
    parser.add_argument("--metrics-dir", default="reports/metrics/generalization")
    parser.add_argument("--figures-dir", default="reports/figures/generalization")
    parser.add_argument("--models", nargs="*", default=["random_forest", "hist_gradient_boosting"])
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--skip", nargs="*", default=[], choices=["lofo", "day", "port"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    metrics_dir, figures_dir = Path(args.metrics_dir), Path(args.figures_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.data)
    feats = feature_columns(df)
    results: dict = {}

    if "lofo" not in args.skip:
        lofo = [r for m in args.models for r in leave_one_family_out(df, feats, m, args.rf_trees)]
        results["leave_one_family_out"] = lofo
        table = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in lofo])
        table.to_csv(metrics_dir / "leave_one_family_out.csv", index=False)
        plot_lofo(table, figures_dir / "leave_one_family_out.png")
    if "day" not in args.skip:
        results["held_out_day"] = [held_out_day(df, feats, m, args.rf_trees) for m in args.models]
    if "port" not in args.skip:
        results["port_ablation"] = port_ablation(df, feats, args.rf_trees)

    (metrics_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
