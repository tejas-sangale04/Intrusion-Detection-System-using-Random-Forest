"""Phase 2: build the cleaned, de-duplicated dataset and the train/val/test split.

Run from the project root (after placing the raw files in data/raw):

    python -m src.data.preprocess

Steps, in order, and why:

1. **Read every raw file in chunks** with the same discovery, encoding and
   header rules as Phase 1 (``src.data.io``), so both phases see identical data.
2. **Normalise labels** (``src.data.labels``) and add ``family`` and
   ``is_attack``. Unknown labels stop the run.
3. **Keep every feature column, unchanged.** Dropping constant or duplicate
   columns, handling infinities and imputing are *learned* steps, so they live
   in the scikit-learn pipeline (``src.features.feature_pipeline``) and are
   fitted on the training split only.
4. **Remove exact duplicate rows** (same features and same label), keeping the
   first occurrence. Phase 1 found 10.9% redundant copies, concentrated in a
   few attacks; left in, a random split puts copies of one flow in both train
   and test and the test score partly measures memorisation. Rows with the same
   features but *different* labels are kept: that ambiguity is real and the
   evaluation should feel it.
5. **Store features as float32.** Halves memory (about 0.8 GB instead of
   1.6 GB for the full dataset) so training fits on an 8 GB laptop. Duplicate
   detection uses the original float64 values, before the cast.
6. **Assign a stratified 70/15/15 train/validation/test split** on the
   de-duplicated rows, stratified by fine label so even Heartbleed (11 rows)
   appears in every part. The split is saved as a column, not as copies of
   the data, so later phases can also build day- or family-based splits from
   the same file.

Outputs (git-ignored, regenerable):
    data/processed/flows.parquet           features + label, family, is_attack, source_file, split
    reports/metrics/preprocessing/summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from src.data.io import discover_files, find_label_column, iter_chunks, normalize_columns, read_raw_header, scan_file
from src.data.labels import normalize_labels

logger = logging.getLogger(__name__)

META_COLUMNS = ["label", "family", "is_attack", "source_file", "split"]
SPLITS = ("train", "val", "test")
DEFAULT_SEED = 42


def build_clean_dataset(raw_dir: str | Path, out_path: str | Path, chunksize: int = 200_000) -> dict:
    """Stream raw files into one de-duplicated parquet file (without split column)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.parquet")

    seen: set[int] = set()
    writer: pq.ParquetWriter | None = None
    feature_cols: list[str] | None = None
    per_file: dict[str, dict[str, int]] = {}
    removed_by_label: dict[str, int] = {}

    try:
        for source in discover_files(raw_dir):
            scan = scan_file(source)
            names = normalize_columns(read_raw_header(source, scan.encoding)).names
            label_col = find_label_column(names)
            file_features = [c for c in names if c != label_col]
            if feature_cols is None:
                feature_cols = file_features
            elif set(file_features) != set(feature_cols):
                raise ValueError(f"{source.name} has a different feature set from the first file")
            stats = per_file.setdefault(source.name, {"rows_in": 0, "rows_out": 0})

            for chunk in iter_chunks(source, scan.encoding, names, chunksize):
                features = chunk[feature_cols].apply(pd.to_numeric, errors="coerce").astype("float64")
                meta = normalize_labels(chunk[label_col])

                row_hash = pd.util.hash_pandas_object(
                    features.assign(_label=meta["label"].to_numpy()), index=False
                ).to_numpy()
                keep = np.zeros(len(row_hash), dtype=bool)
                for i, h in enumerate(row_hash.tolist()):
                    if h not in seen:
                        seen.add(h)
                        keep[i] = True
                for label, n in meta.loc[~keep, "label"].value_counts().items():
                    removed_by_label[label] = removed_by_label.get(label, 0) + int(n)

                out = features.loc[keep].astype("float32")
                out = pd.concat([out, meta.loc[keep]], axis=1)
                out["source_file"] = source.name
                table = pa.Table.from_pandas(out, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(tmp_path, table.schema, compression="zstd")
                writer.write_table(table.cast(writer.schema))
                stats["rows_in"] += len(chunk)
                stats["rows_out"] += int(keep.sum())
            logger.info("%s: %d rows in, %d kept", source.name, stats["rows_in"], stats["rows_out"])
    finally:
        if writer is not None:
            writer.close()

    tmp_path.replace(out_path)
    return {
        "feature_columns": feature_cols,
        "per_file": per_file,
        "rows_in": sum(s["rows_in"] for s in per_file.values()),
        "rows_out": sum(s["rows_out"] for s in per_file.values()),
        "duplicates_removed_by_label": dict(sorted(removed_by_label.items(), key=lambda kv: -kv[1])),
    }


def assign_split(labels: pd.Series, seed: int = DEFAULT_SEED, val: float = 0.15, test: float = 0.15) -> np.ndarray:
    """Stratified train/val/test assignment; returns an array of split names."""
    idx = np.arange(len(labels))
    train_idx, rest_idx = train_test_split(idx, test_size=val + test, stratify=labels, random_state=seed)
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=test / (val + test), stratify=labels.iloc[rest_idx], random_state=seed
    )
    split = np.empty(len(labels), dtype=object)
    split[train_idx], split[val_idx], split[test_idx] = "train", "val", "test"
    return split


def add_split_column(path: str | Path, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Read the clean parquet, add the ``split`` column, rewrite it, return label x split counts."""
    path = Path(path)
    table = pq.read_table(path)
    labels = table.column("label").to_pandas()
    split = assign_split(labels, seed)
    table = table.append_column("split", pa.array(split, type=pa.string()))
    tmp = path.with_suffix(".tmp.parquet")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)
    return pd.crosstab(labels, pd.Series(split, name="split"))[list(SPLITS)]


def load_split(path: str | Path, split: str | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    """Load the processed dataset, optionally a single split, without reading other splits into memory."""
    filters = [("split", "==", split)] if split else None
    return pd.read_parquet(path, columns=columns, filters=filters)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLUMNS]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the cleaned, de-duplicated, split dataset.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="data/processed/flows.parquet")
    parser.add_argument("--metrics-dir", default="reports/metrics/preprocessing")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    summary = build_clean_dataset(args.raw_dir, args.out, args.chunksize)
    counts = add_split_column(args.out, args.seed)
    summary["seed"] = args.seed
    summary["split_rows"] = {s: int(counts[s].sum()) for s in SPLITS}

    metrics = Path(args.metrics_dir)
    metrics.mkdir(parents=True, exist_ok=True)
    counts.to_csv(metrics / "label_by_split.csv")
    (metrics / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("rows in %d, after de-duplication %d; split %s", summary["rows_in"], summary["rows_out"], summary["split_rows"])


if __name__ == "__main__":
    main()
