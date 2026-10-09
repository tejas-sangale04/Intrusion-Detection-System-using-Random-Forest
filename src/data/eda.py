"""Phase 1: exploratory plots, correlation analysis and the inspection report.

Run after ``inspect_dataset`` (it reads that script's outputs, so plots can be
regenerated without rescanning several GB of CSV):

    python -m src.data.eda

Every number in the generated report comes from the inspection outputs. The
report states facts only; interpretation is written by hand in the README so
it is never mistaken for a measured result.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BENIGN = "BENIGN"
CORRELATION_THRESHOLD = 0.95

# Features with a clear traffic meaning, plotted when present. Names are
# matched after whitespace normalisation; missing ones are skipped, never
# assumed.
PREFERRED_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Fwd Packet Length Max",
    "Bwd Packet Length Mean",
    "Average Packet Size",
    "Init_Win_bytes_forward",
    "Destination Port",
]


def signed_log10(x: np.ndarray) -> np.ndarray:
    """sign(x) * log10(1 + |x|): keeps zero and negatives, compresses heavy tails."""
    return np.sign(x) * np.log10(1 + np.abs(x))


def _is_benign(labels: pd.Series) -> pd.Series:
    return labels.astype(str).str.strip().str.upper() == BENIGN


def plot_class_distribution(labels: pd.DataFrame, out: Path) -> None:
    totals = labels["total"].sort_values()
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(totals) + 1.5))
    colors = ["#4C78A8" if str(l).strip().upper() == BENIGN else "#E45756" for l in totals.index]
    ax.barh([str(l) for l in totals.index], totals.to_numpy(), color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("Rows (log scale)")
    ax.set_title("CIC-IDS2017 label distribution (all files)")
    for i, v in enumerate(totals.to_numpy()):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8)
    ax.margins(x=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_label_by_file(labels: pd.DataFrame, out: Path) -> None:
    matrix = labels.drop(columns=["total", "pct_of_rows"])
    short = [c.replace(".pcap_ISCX", "").replace(".csv", "") for c in matrix.columns]
    data = np.log10(matrix.to_numpy(dtype=float) + 1)
    fig, ax = plt.subplots(figsize=(1.1 * len(short) + 4, 0.4 * len(matrix) + 2))
    im = ax.imshow(data, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(short)), short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix)), [str(l) for l in matrix.index], fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = int(matrix.iat[i, j])
            if v:
                ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=6,
                        color="white" if data[i, j] > data.max() * 0.6 else "black")
    fig.colorbar(im, ax=ax, label="log10(rows + 1)")
    ax.set_title("Which file contains which label")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def choose_features(columns: list[str], stats: pd.DataFrame, n: int = 12) -> list[str]:
    chosen = [f for f in PREFERRED_FEATURES if f in columns and not stats.loc[f, "constant"]]
    if len(chosen) < n:
        rest = stats.drop(index=chosen)
        rest = rest[~rest["constant"]].sort_values("std", ascending=False)
        chosen += rest.index[: n - len(chosen)].tolist()
    return chosen[:n]


def plot_feature_distributions(sample: pd.DataFrame, features: list[str], out: Path) -> None:
    benign = _is_benign(sample["Label"])
    cols = 3
    rows = int(np.ceil(len(features) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.9))
    for ax, feat in zip(np.ravel(axes), features, strict=False):
        values = sample[feat].to_numpy(dtype=float)
        finite = np.isfinite(values)
        transformed = signed_log10(values)
        bins = np.histogram_bin_edges(transformed[finite], bins=40) if finite.any() else 10
        for mask, name, color in ((benign, "BENIGN", "#4C78A8"), (~benign, "attack", "#E45756")):
            vals = transformed[mask.to_numpy() & finite]
            if len(vals):
                ax.hist(vals, bins=bins, alpha=0.55, density=True, label=name, color=color)
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel("sign(x)·log10(1+|x|)", fontsize=7)
        ax.tick_params(labelsize=7)
    for ax in np.ravel(axes)[len(features):]:
        ax.axis("off")
    np.ravel(axes)[0].legend(fontsize=7)
    fig.suptitle("Feature distributions, BENIGN vs attack (stratified sample, density-normalised)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def correlation_analysis(sample: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pearson correlation on the uniform sample, with inf treated as missing.

    Constant columns are excluded because their correlation is undefined.
    Returns the matrix and the list of pairs with |r| >= the threshold.
    """
    data = sample[features].replace([np.inf, -np.inf], np.nan)
    data = data.loc[:, data.nunique(dropna=True) > 1]
    corr = data.corr(method="pearson", min_periods=30)
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().rename("pearson_r").reset_index()
    pairs.columns = ["feature_a", "feature_b", "pearson_r"]
    pairs = pairs[pairs["pearson_r"].abs() >= CORRELATION_THRESHOLD]
    pairs = pairs.reindex(pairs["pearson_r"].abs().sort_values(ascending=False).index)
    return corr, pairs.reset_index(drop=True)


def plot_correlation(corr: pd.DataFrame, out: Path) -> None:
    n = len(corr)
    fig, ax = plt.subplots(figsize=(0.16 * n + 4, 0.16 * n + 3))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=90, fontsize=5)
    ax.set_yticks(range(n), corr.index, fontsize=5)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Pearson r")
    ax.set_title("Feature correlation (uniform sample, constant columns excluded)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:,.4g}")
            else:
                cells.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(metrics_dir: Path, figures_dir: Path, report_path: Path, pairs: pd.DataFrame,
                 features_plotted: list[str], sample_sizes: dict[str, int]) -> str:
    summary = json.loads((metrics_dir / "summary.json").read_text(encoding="utf-8"))
    files = pd.read_csv(metrics_dir / "files.csv", keep_default_na=False)
    labels = pd.read_csv(metrics_dir / "label_distribution.csv", index_col="label", keep_default_na=False)
    stats = pd.read_csv(metrics_dir / "column_stats.csv", index_col="column")
    dup_label = pd.read_csv(metrics_dir / "duplicates_by_label.csv", keep_default_na=False)
    dup = summary["duplicates"]
    rel = lambda p: Path(os.path.relpath(p, report_path.parent)).as_posix()  # noqa: E731

    benign_rows = int(labels.loc[[l for l in labels.index if l.strip().upper() == BENIGN], "total"].sum())
    total = summary["total_rows"]
    nonfinite = stats[(stats["nan"] + stats["pos_inf"] + stats["neg_inf"] + stats["unparseable"]) > 0]

    out = [
        "# CIC-IDS2017 dataset inspection",
        "",
        "Generated by `python -m src.data.inspect_dataset` and `python -m src.data.eda`. "
        "Every number below was measured from the files in "
        f"`{summary['raw_dir']}`; nothing is copied from papers.",
        "",
        "## Overview",
        "",
        f"* Files: **{summary['file_count']}**",
        f"* Rows: **{total:,}**",
        f"* Columns: **{summary['column_count']}** ({summary['feature_count']} features + "
        f"`{summary['label_column']}`)",
        f"* Distinct labels: **{summary['label_count']}**",
        f"* BENIGN rows: **{benign_rows:,}** ({100 * benign_rows / total:.2f}%); "
        f"attack rows: **{total - benign_rows:,}** ({100 * (total - benign_rows) / total:.2f}%)",
        f"* Lines the CSV parser could not read: **{summary['unparsed_lines_total']:,}**; "
        f"completely empty rows: **{summary['all_empty_rows_total']:,}**",
        "",
        "## Files",
        "",
        _md_table(files[["file", "encoding", "rows", "columns", "columns_with_whitespace",
                         "duplicate_column_names", "labels", "unparsed_lines"]]),
        "",
        "Header differences between files: "
        + ("none (all files share the same normalised columns in the same order)."
           if not summary["header_differences"] else f"`{summary['header_differences']}`"),
        "",
        "## Labels",
        "",
        f"![Class distribution]({rel(figures_dir / 'class_distribution.png')})",
        "",
        _md_table(labels[["total", "pct_of_rows"]].reset_index()),
        "",
        "Non-ASCII label strings (an encoding artefact to normalise in Phase 2): "
        + (", ".join(f"`{l!r}`" for l in summary["non_ascii_labels"]) or "none."),
        "",
        f"![Labels by file]({rel(figures_dir / 'label_by_file.png')})",
        "",
        "## Missing, infinite and unparseable values",
        "",
        (_md_table(nonfinite[["nan", "pos_inf", "neg_inf", "unparseable"]].reset_index())
         if len(nonfinite) else "No missing, infinite or unparseable values."),
        "",
        "Per-file counts: `reports/metrics/inspection/nonfinite_by_file.csv`.",
        "",
        "## Duplicates",
        "",
        f"* Redundant copies of exact duplicate rows (features and label equal): **{dup['redundant_copies']:,}** "
        f"({100 * dup['redundant_copies'] / total:.2f}% of rows)",
        f"  * within the same file: {dup['redundant_copies_within_files']:,}; "
        f"across files: {dup['redundant_copies_across_files']:,}",
        f"* Unique feature vectors: {dup['unique_feature_vectors']:,}",
        f"* Feature vectors that appear with more than one label: **{dup['conflicting_feature_vectors']:,}** "
        f"(covering {dup['rows_in_conflicting_vectors']:,} rows)",
        "",
        _md_table(dup_label.sort_values("rows", ascending=False)),
        "",
        "Label sets of conflicting vectors (top 10):",
        "",
        _md_table(pd.DataFrame(list(dup["conflicting_label_sets"].items())[:10],
                               columns=["labels", "feature_vectors"]))
        if dup["conflicting_label_sets"] else "None.",
        "",
        "## Unusable or redundant columns",
        "",
        "* Constant columns (one finite value everywhere): "
        + (", ".join(f"`{c}`" for c in summary["constant_columns"]) or "none"),
        "* Groups of columns with identical contents: "
        + ("; ".join(" = ".join(f"`{c}`" for c in g) for g in summary["identical_column_groups"]) or "none"),
        "* Columns that look like identifiers or context (to decide on in Phase 2): "
        + (", ".join(f"`{c}` ({k})" for c, k in summary["identifier_columns"].items()) or "none"),
        "",
        "## Numeric summaries",
        "",
        "Full table: `reports/metrics/inspection/column_stats.csv` (finite values only for mean/std/min/max).",
        "",
        _md_table(stats.loc[[f for f in features_plotted if f in stats.index],
                            ["mean", "std", "min", "max", "zeros", "negatives"]].reset_index()),
        "",
        f"![Feature distributions]({rel(figures_dir / 'feature_distributions.png')})",
        "",
        "## Correlation",
        "",
        f"Pearson correlation on a uniform random sample of {sample_sizes['uniform']:,} rows, "
        f"with infinities treated as missing. Pairs with |r| >= {CORRELATION_THRESHOLD}: **{len(pairs)}**.",
        "",
        f"![Correlation]({rel(figures_dir / 'correlation_heatmap.png')})",
        "",
        _md_table(pairs.head(25)) if len(pairs) else "",
        "",
        "## Sampling notes",
        "",
        f"* Distribution plots use a stratified sample of up to {summary['settings']['per_label']:,} rows per label "
        f"({sample_sizes['stratified']:,} rows). It shows rare attacks but does not reflect real class proportions.",
        f"* Correlations use a uniform sample of {sample_sizes['uniform']:,} rows, which keeps the real class mix. "
        "Very rare labels therefore barely influence it.",
        f"* Both samples are exact uniform draws (bottom-k sampling, seed {summary['settings']['seed']}); all counts "
        "above (rows, labels, missing values, duplicates) use every row, not a sample.",
        "",
    ]
    text = "\n".join(out)
    report_path.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plots and report from the inspection outputs.")
    parser.add_argument("--metrics-dir", default="reports/metrics/inspection")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--figures-dir", default="reports/figures/inspection")
    parser.add_argument("--report", default="reports/dataset_inspection.md")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    metrics_dir, figures_dir, processed = Path(args.metrics_dir), Path(args.figures_dir), Path(args.processed_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(metrics_dir / "label_distribution.csv", index_col="label", keep_default_na=False)
    stats = pd.read_csv(metrics_dir / "column_stats.csv", index_col="column")
    uniform = pd.read_parquet(processed / "eda_sample_uniform.parquet")
    stratified = pd.read_parquet(processed / "eda_sample_stratified.parquet")
    features = stats.index.tolist()

    plot_class_distribution(labels, figures_dir / "class_distribution.png")
    plot_label_by_file(labels, figures_dir / "label_by_file.png")
    chosen = choose_features(features, stats)
    plot_feature_distributions(stratified, chosen, figures_dir / "feature_distributions.png")
    corr, pairs = correlation_analysis(uniform, features)
    plot_correlation(corr, figures_dir / "correlation_heatmap.png")
    pairs.to_csv(metrics_dir / "correlated_pairs.csv", index=False)

    build_report(metrics_dir, figures_dir, Path(args.report), pairs, chosen,
                 {"uniform": len(uniform), "stratified": len(stratified)})
    logger.info("wrote report -> %s", args.report)


if __name__ == "__main__":
    main()
