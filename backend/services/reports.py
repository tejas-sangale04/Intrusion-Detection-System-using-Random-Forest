"""Reads the offline evaluation saved by Phases 3 to 6, for ``/metrics``.

Paths in the response use forward slashes on every OS, so clients and tests
see the same strings on Windows and Linux.

The API never computes or estimates these numbers: it returns what the
training scripts wrote, plus the path each value came from. A file that does
not exist yet is reported as missing instead of being filled with a default.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

SOURCES = {
    "classifier_baselines": "baselines/results.json",
    "generalization": "generalization/leave_one_family_out.csv",
    "anomaly_and_hybrid": "anomaly/results.json",
    "hybrid_held_out_families": "anomaly/hybrid_held_out_families.csv",
    "explainability": "explainability/summary.json",
    "global_feature_importance": "explainability/global_importance.csv",
}


def _read(path: Path):
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        for k, v in row.items():
            try:
                row[k] = float(v) if "." in v or "e" in v.lower() else int(v)
            except (ValueError, AttributeError):
                pass
    return rows


def offline_metrics(reports_dir: Path) -> dict:
    out, missing = {}, []
    for key, rel in SOURCES.items():
        path = Path(reports_dir) / rel
        if path.exists():
            out[key] = {"source": path.as_posix(), "data": _read(path)}
        else:
            out[key] = None
            missing.append(path.as_posix())
    if out["classifier_baselines"] is not None:
        # Only the classifier the API serves; the full comparison stays in the file.
        data = out["classifier_baselines"]["data"]
        out["classifier_baselines"]["data"] = {
            "served_model": "random_forest",
            "random_forest": data.get("results", {}).get("random_forest"),
        }
    out["missing_files"] = missing
    return out
