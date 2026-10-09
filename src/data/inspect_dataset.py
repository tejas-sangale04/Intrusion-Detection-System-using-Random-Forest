"""Phase 1: read-only, chunked inspection of the CIC-IDS2017 CSV files.

Run from the project root:

    python -m src.data.inspect_dataset --raw-dir data/raw

What it measures (all from the actual files, nothing assumed):

* files, encodings, rows, columns and header differences between files
* label distribution per file and overall (raw label strings, unmodified)
* missing values, +inf / -inf and unparseable values per column and file
* exact duplicate rows (within a file and across files) and "conflicting"
  duplicates: identical feature vectors that carry different labels
* constant columns and columns that are exact copies of other columns
* streaming numeric summaries (count, mean, std, min, max, zeros, negatives)
* columns that look like identifiers (IPs, ports, timestamps, flow IDs)

It also keeps two random samples for EDA (``eda.py``), drawn with bottom-k
sampling so each is an exact uniform sample without loading the data at once:

* ``sample_uniform``: uniform over all rows, used for correlations and
  approximate quantiles because it preserves the real class mix
* ``sample_stratified``: up to ``--per-label`` rows of every label, used for
  per-class plots so rare attacks are visible. It does NOT reflect real class
  proportions and must never be used to estimate them.

Memory stays bounded by the chunk size plus the samples plus three small
arrays per row (two 64-bit hashes and a label code) for duplicate analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.io import (
    DataSource,
    discover_files,
    find_label_column,
    iter_chunks,
    normalize_columns,
    read_raw_header,
    scan_file,
)

logger = logging.getLogger(__name__)

DEFAULT_CHUNKSIZE = 200_000
DEFAULT_PER_LABEL = 5_000
DEFAULT_UNIFORM = 100_000
DEFAULT_SEED = 42

# Columns whose names suggest they identify a host, connection or moment in
# time rather than describe traffic behaviour. They are only *flagged* here;
# Phase 2 decides what to drop and documents why.
IDENTIFIER_PATTERNS = {
    "flow id": re.compile(r"^flow[ _]?id$", re.I),
    "ip address": re.compile(r"(^|[ _])(src|source|dst|destination)[ _]?ip$", re.I),
    "port": re.compile(r"(^|[ _])(src|source|dst|destination)[ _]?port$", re.I),
    "timestamp": re.compile(r"^time[ _]?stamp$", re.I),
    "protocol": re.compile(r"^protocol$", re.I),
}


@dataclass
class NumericAccumulator:
    """Streaming per-column statistics, merged chunk by chunk.

    Mean and variance use Chan et al.'s parallel update, which is numerically
    stable, unlike the naive sum / sum-of-squares formula that loses precision
    on columns such as ``Flow Bytes/s`` with values spanning many magnitudes.
    """

    columns: list[str]
    n: np.ndarray = field(init=False)
    mean: np.ndarray = field(init=False)
    m2: np.ndarray = field(init=False)
    min: np.ndarray = field(init=False)
    max: np.ndarray = field(init=False)
    nan: np.ndarray = field(init=False)
    pos_inf: np.ndarray = field(init=False)
    neg_inf: np.ndarray = field(init=False)
    zeros: np.ndarray = field(init=False)
    negatives: np.ndarray = field(init=False)
    fingerprint: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        k = len(self.columns)
        self.n = np.zeros(k, dtype=np.int64)
        self.mean = np.zeros(k)
        self.m2 = np.zeros(k)
        self.min = np.full(k, np.inf)
        self.max = np.full(k, -np.inf)
        for name in ("nan", "pos_inf", "neg_inf", "zeros", "negatives"):
            setattr(self, name, np.zeros(k, dtype=np.int64))
        self.fingerprint = np.zeros(k, dtype=np.uint64)

    def update(self, values: np.ndarray, row_offset: int) -> None:
        """``values`` is a float64 array (rows x columns) for one chunk."""
        finite = np.isfinite(values)
        self.nan += np.isnan(values).sum(axis=0)
        self.pos_inf += np.isposinf(values).sum(axis=0)
        self.neg_inf += np.isneginf(values).sum(axis=0)
        self.zeros += (values == 0).sum(axis=0)
        self.negatives += (finite & (values < 0)).sum(axis=0)

        n_c = finite.sum(axis=0)
        with np.errstate(invalid="ignore"):
            masked = np.where(finite, values, np.nan)
            self.min = np.fmin(self.min, np.nanmin(np.where(finite, values, np.inf), axis=0))
            self.max = np.fmax(self.max, np.nanmax(np.where(finite, values, -np.inf), axis=0))
            mean_c = np.where(n_c > 0, np.nansum(masked, axis=0) / np.maximum(n_c, 1), 0.0)
            m2_c = np.nansum((masked - mean_c) ** 2, axis=0)
        total = self.n + n_c
        delta = mean_c - self.mean
        safe_total = np.maximum(total, 1)
        self.mean = self.mean + delta * n_c / safe_total
        self.m2 = self.m2 + m2_c + delta**2 * self.n * n_c / safe_total
        self.n = total

        # Order-sensitive fingerprint of each column's full contents. Two
        # columns with equal fingerprints are, up to a 2^-64 collision chance,
        # identical value for value, which is far cheaper than comparing all
        # column pairs on every chunk.
        rows = pd.util.hash_array(np.arange(row_offset, row_offset + len(values), dtype=np.int64))
        sums = np.array(
            [(pd.util.hash_array(values[:, j]) ^ rows).sum(dtype=np.uint64) for j in range(values.shape[1])],
            dtype=np.uint64,
        )
        self.fingerprint += sums  # array addition wraps modulo 2^64, as intended

    def to_frame(self) -> pd.DataFrame:
        with np.errstate(invalid="ignore"):
            std = np.sqrt(np.where(self.n > 1, self.m2 / np.maximum(self.n - 1, 1), np.nan))
        has = self.n > 0
        return pd.DataFrame(
            {
                "finite_count": self.n,
                "nan": self.nan,
                "pos_inf": self.pos_inf,
                "neg_inf": self.neg_inf,
                "mean": np.where(has, self.mean, np.nan),
                "std": std,
                "min": np.where(has, self.min, np.nan),
                "max": np.where(has, self.max, np.nan),
                "zeros": self.zeros,
                "negatives": self.negatives,
            },
            index=pd.Index(self.columns, name="column"),
        )


class BottomKSampler:
    """Exact uniform sampling without replacement over a stream.

    Every row gets an independent uniform random key; keeping the k smallest
    keys seen so far yields a uniform sample of size k from all rows. With
    ``group`` set, the k smallest keys are kept per group (stratified).
    """

    def __init__(self, k: int, group: str | None, rng: np.random.Generator) -> None:
        self.k, self.group, self.rng = k, group, rng
        self.sample: pd.DataFrame | None = None

    def update(self, chunk: pd.DataFrame) -> None:
        if self.k <= 0:
            return
        chunk = chunk.assign(_key=self.rng.random(len(chunk)))
        pool = chunk if self.sample is None else pd.concat([self.sample, chunk], ignore_index=True)
        pool = pool.sort_values("_key", kind="stable")
        if self.group is None:
            self.sample = pool.head(self.k)
        else:
            self.sample = pool.groupby(self.group, sort=False, dropna=False).head(self.k)

    def result(self) -> pd.DataFrame:
        if self.sample is None:
            return pd.DataFrame()
        return self.sample.drop(columns="_key").reset_index(drop=True)


def _coerce_numeric(chunk: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Convert feature columns to float64 and count values that fail to parse.

    pandas already parses ``Infinity``/``NaN`` text. A column only arrives as
    non-numeric when some cell is genuinely malformed; those cells become NaN
    here but are counted separately so they are not mistaken for missing data.
    """
    unparseable = pd.Series(0, index=columns, dtype=np.int64)
    out = {}
    for col in columns:
        s = chunk[col]
        if not pd.api.types.is_numeric_dtype(s):
            converted = pd.to_numeric(s, errors="coerce")
            unparseable[col] = int((converted.isna() & s.notna()).sum())
            s = converted
        out[col] = s.astype("float64")
    return pd.DataFrame(out, index=chunk.index), unparseable


@dataclass
class FileReport:
    name: str
    location: str
    encoding: str
    encoding_notes: list[str]
    columns: list[str]
    raw_columns: list[str]
    stripped_columns: int
    deduplicated_columns: list[tuple[str, str]]
    label_column: str
    data_lines: int
    rows: int = 0
    all_empty_rows: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def unparsed_lines(self) -> int:
        return self.data_lines - self.rows


def inspect(
    raw_dir: str | Path,
    chunksize: int = DEFAULT_CHUNKSIZE,
    per_label: int = DEFAULT_PER_LABEL,
    uniform_size: int = DEFAULT_UNIFORM,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Inspect every CSV under ``raw_dir`` and return all results in memory."""
    sources = discover_files(raw_dir)
    logger.info("Found %d CSV file(s) under %s", len(sources), raw_dir)

    rng = np.random.default_rng(seed)
    uniform = BottomKSampler(uniform_size, None, rng)
    stratified = BottomKSampler(per_label, "Label", rng)

    reference_cols: list[str] | None = None
    feature_cols: list[str] | None = None
    accumulator: NumericAccumulator | None = None
    per_file_missing: dict[str, pd.DataFrame] = {}
    files: list[FileReport] = []
    label_codes: dict[str, int] = {}
    hash_parts: list[pd.DataFrame] = []
    unparseable_total: pd.Series | None = None
    header_differences: dict[str, dict] = {}
    rows_seen = 0

    for file_idx, source in enumerate(sources):
        started = time.perf_counter()
        scan = scan_file(source)
        raw_header = read_raw_header(source, scan.encoding)
        norm = normalize_columns(raw_header)
        label_col = find_label_column(norm.names)
        report = FileReport(
            name=source.name,
            location=source.location,
            encoding=scan.encoding,
            encoding_notes=scan.notes,
            columns=norm.names,
            raw_columns=raw_header,
            stripped_columns=len(norm.stripped),
            deduplicated_columns=norm.deduplicated,
            label_column=label_col,
            data_lines=max(scan.line_count - 1, 0),
        )
        logger.info("Inspecting %s (%s, %d data lines)", source.name, scan.encoding, report.data_lines)

        file_features = [c for c in norm.names if c != label_col]
        if reference_cols is None:
            reference_cols, feature_cols = norm.names, file_features
            accumulator = NumericAccumulator(feature_cols)
            unparseable_total = pd.Series(0, index=feature_cols, dtype=np.int64)
        elif norm.names != reference_cols:
            header_differences[source.name] = {
                "missing": [c for c in reference_cols if c not in norm.names],
                "extra": [c for c in norm.names if c not in reference_cols],
                "same_set_different_order": set(norm.names) == set(reference_cols),
            }
            missing = header_differences[source.name]["missing"]
            extra = header_differences[source.name]["extra"]
            if missing or extra:
                raise ValueError(
                    f"{source.name} has a different schema from {sources[0].name}: "
                    f"missing={missing} extra={extra}. Inspect it separately before combining."
                )
        assert feature_cols is not None and accumulator is not None and unparseable_total is not None

        file_stats = NumericAccumulator(feature_cols)
        for chunk in iter_chunks(source, scan.encoding, norm.names, chunksize):
            features, unparseable = _coerce_numeric(chunk, feature_cols)
            unparseable_total += unparseable
            labels = chunk[label_col].astype("object")
            report.all_empty_rows += int((features.isna().all(axis=1) & labels.isna()).sum())

            values = features.to_numpy()
            accumulator.update(values, rows_seen)
            file_stats.update(values, rows_seen)

            label_str = labels.where(labels.notna(), "<missing>").astype(str)
            for label, count in label_str.value_counts(sort=False).items():
                report.label_counts[label] = report.label_counts.get(label, 0) + int(count)
                label_codes.setdefault(label, len(label_codes))

            hash_parts.append(
                pd.DataFrame(
                    {
                        "feature_hash": pd.util.hash_pandas_object(features, index=False).to_numpy(),
                        "label": label_str.map(label_codes).to_numpy(np.int32),
                        "file": np.int16(file_idx),
                    }
                )
            )

            sample_frame = features.assign(Label=label_str.to_numpy(), source_file=source.name)
            uniform.update(sample_frame)
            stratified.update(sample_frame)

            report.rows += len(chunk)
            rows_seen += len(chunk)

        per_file_missing[source.name] = file_stats.to_frame()[["nan", "pos_inf", "neg_inf"]]
        report.seconds = round(time.perf_counter() - started, 2)
        files.append(report)
        logger.info("  %s: %d rows, %d labels, %.1fs", source.name, report.rows, len(report.label_counts), report.seconds)

    assert accumulator is not None and feature_cols is not None and unparseable_total is not None
    column_stats = accumulator.to_frame()
    column_stats["unparseable"] = unparseable_total
    column_stats["constant"] = (
        (column_stats["finite_count"] > 0)
        & (column_stats["min"] == column_stats["max"])
    )

    hashes = pd.concat(hash_parts, ignore_index=True)
    code_to_label = {code: label for label, code in label_codes.items()}

    return {
        "raw_dir": str(raw_dir),
        "files": files,
        "feature_columns": feature_cols,
        "label_column": files[0].label_column,
        "header_differences": header_differences,
        "column_stats": column_stats,
        "missing_by_file": per_file_missing,
        "identical_columns": _identical_columns(feature_cols, accumulator.fingerprint),
        "identifier_columns": _identifier_columns(feature_cols),
        "duplicates": _duplicate_analysis(hashes, code_to_label, [f.name for f in files]),
        "sample_uniform": uniform.result(),
        "sample_stratified": stratified.result(),
        "settings": {"chunksize": chunksize, "per_label": per_label, "uniform_size": uniform_size, "seed": seed},
    }


def _identical_columns(columns: list[str], fingerprints: np.ndarray) -> list[list[str]]:
    groups: dict[int, list[str]] = {}
    for col, fp in zip(columns, fingerprints, strict=True):
        groups.setdefault(int(fp), []).append(col)
    return [cols for cols in groups.values() if len(cols) > 1]


def _identifier_columns(columns: list[str]) -> dict[str, str]:
    found = {}
    for col in columns:
        for kind, pattern in IDENTIFIER_PATTERNS.items():
            if pattern.search(col.strip()):
                found[col] = kind
    return found


def _duplicate_analysis(hashes: pd.DataFrame, code_to_label: dict[int, str], file_names: list[str]) -> dict:
    """Quantify exact duplicates, which matter for leakage.

    If identical rows sit on both sides of a random train/test split, the test
    set partly measures memorisation. Conflicting duplicates (same features,
    different labels) cap the accuracy any model can reach on those rows.
    Hash equality is used as row equality; with 64-bit hashes the chance of a
    false match among a few million rows is negligible (~1e-7).
    """
    n = len(hashes)
    exact = hashes.duplicated(["feature_hash", "label"])
    within = hashes.duplicated(["file", "feature_hash", "label"])
    feature_dup = hashes.duplicated(["feature_hash"])

    by_label = (
        pd.DataFrame({"label": hashes["label"].map(code_to_label), "redundant": exact})
        .groupby("label")["redundant"]
        .agg(rows="size", redundant_copies="sum")
    )
    by_label["redundant_pct"] = (100 * by_label["redundant_copies"] / by_label["rows"]).round(2)

    by_file = (
        pd.DataFrame({"file": hashes["file"].map(dict(enumerate(file_names))), "redundant": within})
        .groupby("file")["redundant"]
        .agg(rows="size", within_file_redundant_copies="sum")
    )

    labels_per_vector = hashes.groupby("feature_hash")["label"].nunique()
    conflicting = labels_per_vector[labels_per_vector > 1]
    conflict_rows = hashes[hashes["feature_hash"].isin(conflicting.index)]
    pair_counts: dict[str, int] = {}
    for _, labels in conflict_rows.groupby("feature_hash")["label"]:
        key = " | ".join(sorted(code_to_label[c] for c in labels.unique()))
        pair_counts[key] = pair_counts.get(key, 0) + 1

    return {
        "rows": int(n),
        "unique_rows": int(n - exact.sum()),
        "redundant_copies": int(exact.sum()),
        "redundant_copies_within_files": int(within.sum()),
        "redundant_copies_across_files": int(exact.sum() - within.sum()),
        "unique_feature_vectors": int(n - feature_dup.sum()),
        "conflicting_feature_vectors": int(len(conflicting)),
        "rows_in_conflicting_vectors": int(len(conflict_rows)),
        "conflicting_label_sets": dict(sorted(pair_counts.items(), key=lambda kv: -kv[1])),
        "by_label": by_label.reset_index(),
        "by_file": by_file.reset_index(),
    }


def write_outputs(result: dict, metrics_dir: str | Path, processed_dir: str | Path) -> dict[str, Path]:
    """Persist inspection results as CSV/JSON tables and parquet samples."""
    metrics_dir = Path(metrics_dir)
    processed_dir = Path(processed_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    files: list[FileReport] = result["files"]
    files_table = pd.DataFrame(
        [
            {
                "file": f.name,
                "encoding": f.encoding,
                "rows": f.rows,
                "data_lines": f.data_lines,
                "unparsed_lines": f.unparsed_lines,
                "all_empty_rows": f.all_empty_rows,
                "columns": len(f.columns),
                "columns_with_whitespace": f.stripped_columns,
                "duplicate_column_names": "; ".join(f"{a} -> {b}" for a, b in f.deduplicated_columns),
                "labels": len(f.label_counts),
                "seconds": f.seconds,
            }
            for f in files
        ]
    )
    paths["files"] = metrics_dir / "files.csv"
    files_table.to_csv(paths["files"], index=False)

    labels = pd.DataFrame({f.name: pd.Series(f.label_counts, dtype="int64") for f in files}).fillna(0).astype("int64")
    labels.index.name = "label"
    labels["total"] = labels.sum(axis=1)
    labels = labels.sort_values("total", ascending=False)
    labels["pct_of_rows"] = (100 * labels["total"] / labels["total"].sum()).round(4)
    paths["label_distribution"] = metrics_dir / "label_distribution.csv"
    labels.to_csv(paths["label_distribution"])

    paths["column_stats"] = metrics_dir / "column_stats.csv"
    result["column_stats"].to_csv(paths["column_stats"])

    missing_rows = []
    for fname, frame in result["missing_by_file"].items():
        nonzero = frame[(frame != 0).any(axis=1)]
        for col, row in nonzero.iterrows():
            missing_rows.append({"file": fname, "column": col, **{k: int(v) for k, v in row.items()}})
    paths["nonfinite_by_file"] = metrics_dir / "nonfinite_by_file.csv"
    pd.DataFrame(missing_rows, columns=["file", "column", "nan", "pos_inf", "neg_inf"]).to_csv(
        paths["nonfinite_by_file"], index=False
    )

    dup = dict(result["duplicates"])
    paths["duplicates_by_label"] = metrics_dir / "duplicates_by_label.csv"
    dup.pop("by_label").to_csv(paths["duplicates_by_label"], index=False)
    paths["duplicates_by_file"] = metrics_dir / "duplicates_by_file.csv"
    dup.pop("by_file").to_csv(paths["duplicates_by_file"], index=False)

    stats = result["column_stats"]
    summary = {
        "raw_dir": result["raw_dir"],
        "file_count": len(files),
        "total_rows": int(sum(f.rows for f in files)),
        "column_count": len(files[0].columns),
        "feature_count": len(result["feature_columns"]),
        "label_column": result["label_column"],
        "label_count": int(len(labels)),
        "encodings": {f.name: f.encoding for f in files},
        "header_differences": result["header_differences"],
        "unparsed_lines_total": int(sum(f.unparsed_lines for f in files)),
        "all_empty_rows_total": int(sum(f.all_empty_rows for f in files)),
        "columns_with_nan": stats.index[stats["nan"] > 0].tolist(),
        "columns_with_inf": stats.index[(stats["pos_inf"] + stats["neg_inf"]) > 0].tolist(),
        "columns_with_unparseable": stats.index[stats["unparseable"] > 0].tolist(),
        "rows_with_nonfinite_note": "per-column counts are in column_stats.csv; a row may be counted in several columns",
        "constant_columns": stats.index[stats["constant"]].tolist(),
        "identical_column_groups": result["identical_columns"],
        "identifier_columns": result["identifier_columns"],
        "non_ascii_labels": [label for label in labels.index if not str(label).isascii()],
        "duplicates": dup,
        "settings": result["settings"],
    }
    paths["summary"] = metrics_dir / "summary.json"
    paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    paths["sample_uniform"] = processed_dir / "eda_sample_uniform.parquet"
    result["sample_uniform"].to_parquet(paths["sample_uniform"], index=False)
    paths["sample_stratified"] = processed_dir / "eda_sample_stratified.parquet"
    result["sample_stratified"].to_parquet(paths["sample_stratified"], index=False)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--metrics-dir", default="reports/metrics/inspection")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--per-label", type=int, default=DEFAULT_PER_LABEL)
    parser.add_argument("--uniform-size", type=int, default=DEFAULT_UNIFORM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = inspect(args.raw_dir, args.chunksize, args.per_label, args.uniform_size, args.seed)
    paths = write_outputs(result, args.metrics_dir, args.processed_dir)
    for name, path in paths.items():
        logger.info("wrote %s -> %s", name, path)


if __name__ == "__main__":
    main()
