from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.labels import LABEL_TO_FAMILY, normalize_label, normalize_labels
from src.data.preprocess import add_split_column, assign_split, build_clean_dataset, feature_columns, load_split


@pytest.mark.parametrize(
    "raw, expected",
    [
        (" BENIGN ", "BENIGN"),
        ("Benign", "BENIGN"),
        ("Web Attack � Brute Force", "Web Attack - Brute Force"),
        ("Web Attack – XSS", "Web Attack - XSS"),
        ("Web Attack  \x96  Sql Injection", "Web Attack - Sql Injection"),
        ("DoS Hulk", "DoS Hulk"),
    ],
)
def test_normalize_label(raw: str, expected: str) -> None:
    assert normalize_label(raw) == expected


def test_unknown_or_missing_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown label"):
        normalize_label("Totally New Attack")
    with pytest.raises(ValueError, match="Missing"):
        normalize_label(np.nan)


def test_normalize_labels_adds_family_and_binary() -> None:
    out = normalize_labels(pd.Series(["BENIGN", "SSH-Patator", "Web Attack � XSS"]))
    assert out["family"].tolist() == ["BENIGN", "Brute Force", "Web Attack"]
    assert out["is_attack"].tolist() == [0, 1, 1]
    assert set(LABEL_TO_FAMILY.values()) >= set(out["family"])


def test_clean_dataset_removes_exact_duplicates_only(raw_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "flows.parquet"
    summary = build_clean_dataset(raw_dir, out, chunksize=2)
    df = pd.read_parquet(out)
    # 8 raw rows: one duplicate within Monday, one across files -> 6 remain.
    assert summary["rows_in"] == 8 and summary["rows_out"] == 6 == len(df)
    assert summary["duplicates_removed_by_label"] == {"BENIGN": 2}
    # Same features, different label (BENIGN vs DoS Hulk) are both kept.
    assert {"BENIGN", "DoS Hulk"} <= set(df["label"])
    assert "Web Attack - Brute Force" in set(df["label"])
    # Features are kept unchanged (no columns dropped here) and inf survives.
    assert "Bwd PSH Flags" in df.columns and "Fwd Header Length.1" in df.columns
    assert np.isinf(df["Flow Packets/s"]).sum() == 3  # two +inf, one -inf
    assert df[feature_columns(df)].dtypes.eq("float32").all()


def test_split_is_stratified_disjoint_and_reproducible() -> None:
    labels = pd.Series(["BENIGN"] * 700 + ["DoS Hulk"] * 200 + ["Heartbleed"] * 20)
    a, b = assign_split(labels, seed=1), assign_split(labels, seed=1)
    assert (a == b).all()
    counts = pd.crosstab(labels, a)
    assert (counts > 0).all().all()  # every label appears in every split
    assert counts.loc["BENIGN", "train"] == 490


def test_split_column_round_trip(raw_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "flows.parquet"
    build_clean_dataset(raw_dir, out)
    # Tiny fixture: replicate rows so every label can be stratified.
    df = pd.read_parquet(out)
    big = pd.concat([df] * 10, ignore_index=True)
    big["Flow Duration"] = np.arange(len(big), dtype="float32")
    big.to_parquet(out, index=False)
    add_split_column(out, seed=0)
    parts = {s: load_split(out, s) for s in ("train", "val", "test")}
    assert sum(len(p) for p in parts.values()) == len(big)
    ids = [set(p["Flow Duration"]) for p in parts.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
