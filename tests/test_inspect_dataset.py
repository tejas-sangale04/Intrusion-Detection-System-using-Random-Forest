from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import eda
from src.data.inspect_dataset import BottomKSampler, NumericAccumulator, inspect, write_outputs
from src.data.io import discover_files, find_label_column, normalize_columns, scan_file


def test_discovers_loose_and_zipped_csvs_but_not_macos_metadata(raw_dir: Path) -> None:
    names = [s.name for s in discover_files(raw_dir)]
    assert names == ["Monday-WorkingHours.pcap_ISCX.csv", "Thursday-WorkingHours.pcap_ISCX.csv"]


def test_extracted_copy_takes_precedence_over_zip_member(raw_dir: Path) -> None:
    extracted = raw_dir / "Thursday-WorkingHours.pcap_ISCX.csv"
    extracted.write_bytes(b"a, Label\n1,BENIGN\n")
    sources = discover_files(raw_dir)
    thursday = next(s for s in sources if s.name == extracted.name)
    assert thursday.member is None


def test_missing_raw_dir_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_files(tmp_path / "nope")


def test_encoding_detection_falls_back_to_cp1252(raw_dir: Path) -> None:
    by_name = {s.name: s for s in discover_files(raw_dir)}
    assert scan_file(by_name["Monday-WorkingHours.pcap_ISCX.csv"]).encoding == "utf-8"
    thursday = scan_file(by_name["Thursday-WorkingHours.pcap_ISCX.csv"])
    assert thursday.encoding == "cp1252"
    assert thursday.line_count == 5


def test_normalize_columns_strips_and_deduplicates() -> None:
    norm = normalize_columns([" Flow Duration", "Fwd Header Length", " Fwd Header Length ", " Label"])
    assert norm.names == ["Flow Duration", "Fwd Header Length", "Fwd Header Length.1", "Label"]
    assert norm.deduplicated == [("Fwd Header Length", "Fwd Header Length.1")]
    assert len(norm.stripped) == 3


def test_label_column_must_exist_exactly_once() -> None:
    assert find_label_column(["a", "label"]) == "label"
    with pytest.raises(ValueError):
        find_label_column(["a", "b"])


def test_inspect_counts_match_hand_built_fixture(raw_dir: Path) -> None:
    result = inspect(raw_dir, chunksize=2, per_label=10, uniform_size=100)
    files = {f.name: f for f in result["files"]}
    assert files["Monday-WorkingHours.pcap_ISCX.csv"].rows == 4
    assert files["Thursday-WorkingHours.pcap_ISCX.csv"].rows == 4
    assert all(f.unparsed_lines == 0 for f in files.values())
    assert files["Thursday-WorkingHours.pcap_ISCX.csv"].label_counts == {
        "BENIGN": 1,
        "DoS Hulk": 1,
        "Web Attack – Brute Force": 1,  # 0x96 decoded as an en dash, label kept as-is
        "SSH-Patator": 1,
    }

    stats = result["column_stats"]
    assert stats.loc["Flow Bytes/s", ["nan", "pos_inf", "unparseable"]].tolist() == [3, 1, 1]  # nan includes the unparseable cell
    assert stats.loc["Flow Packets/s", ["pos_inf", "neg_inf"]].tolist() == [2, 1]
    assert stats.loc["Flow Duration", "min"] == 0 and stats.loc["Flow Duration", "max"] == 200
    assert stats.loc["Flow Duration", "mean"] == pytest.approx(np.mean([100, 100, 200, 0, 100, 200, 5, 7]))
    assert stats.loc["Flow Duration", "std"] == pytest.approx(np.std([100, 100, 200, 0, 100, 200, 5, 7], ddof=1))
    assert stats.index[stats["constant"]].tolist() == ["Bwd PSH Flags"]
    assert ["Fwd Header Length", "Fwd Header Length.1"] in result["identical_columns"]
    assert result["identifier_columns"] == {"Destination Port": "port"}

    dup = result["duplicates"]
    assert dup["rows"] == 8
    assert dup["redundant_copies"] == 2  # one within Monday, one across files
    assert dup["redundant_copies_within_files"] == 1
    assert dup["redundant_copies_across_files"] == 1
    assert dup["conflicting_feature_vectors"] == 1
    assert dup["conflicting_label_sets"] == {"BENIGN | DoS Hulk": 1}


def test_chunk_size_does_not_change_results(raw_dir: Path) -> None:
    small = inspect(raw_dir, chunksize=1, per_label=10, uniform_size=100)
    large = inspect(raw_dir, chunksize=1000, per_label=10, uniform_size=100)
    pd.testing.assert_frame_equal(small["column_stats"], large["column_stats"])
    assert small["identical_columns"] == large["identical_columns"]


def test_streaming_stats_match_numpy_on_random_data() -> None:
    rng = np.random.default_rng(0)
    data = rng.lognormal(10, 3, size=(1000, 3))
    data[rng.random(data.shape) < 0.05] = np.inf
    acc = NumericAccumulator(["a", "b", "c"])
    for start in range(0, 1000, 137):
        acc.update(data[start : start + 137], start)
    frame = acc.to_frame()
    for j, col in enumerate("abc"):
        finite = data[:, j][np.isfinite(data[:, j])]
        assert frame.loc[col, "mean"] == pytest.approx(finite.mean(), rel=1e-9)
        assert frame.loc[col, "std"] == pytest.approx(finite.std(ddof=1), rel=1e-9)


def test_bottom_k_stratified_sampler_keeps_every_rare_row() -> None:
    sampler = BottomKSampler(5, "Label", np.random.default_rng(1))
    sampler.update(pd.DataFrame({"x": range(100), "Label": ["BENIGN"] * 98 + ["Heartbleed"] * 2}))
    out = sampler.result()
    assert out["Label"].value_counts().to_dict() == {"BENIGN": 5, "Heartbleed": 2}


def test_outputs_and_report_are_written(raw_dir: Path, tmp_path: Path) -> None:
    metrics, processed, figures = tmp_path / "m", tmp_path / "p", tmp_path / "f"
    result = inspect(raw_dir, chunksize=3, per_label=10, uniform_size=100)
    write_outputs(result, metrics, processed)
    summary = json.loads((metrics / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_rows"] == 8
    assert summary["non_ascii_labels"] == ["Web Attack – Brute Force"]

    eda.main([
        "--metrics-dir", str(metrics), "--processed-dir", str(processed),
        "--figures-dir", str(figures), "--report", str(tmp_path / "report.md"),
    ])
    for name in ("class_distribution", "label_by_file", "feature_distributions", "correlation_heatmap"):
        assert (figures / f"{name}.png").stat().st_size > 0
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Rows: **8**" in report
    assert "BENIGN \\| DoS Hulk" in report  # pipe escaped inside a markdown table


def test_raw_files_are_not_modified(raw_dir: Path) -> None:
    before = {p: p.read_bytes() for p in raw_dir.iterdir()}
    inspect(raw_dir, chunksize=2, per_label=10, uniform_size=100)
    assert {p: p.read_bytes() for p in raw_dir.iterdir()} == before
