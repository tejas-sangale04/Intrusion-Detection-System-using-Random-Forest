"""Write example request bodies from real CIC-IDS2017 test-split flows.

    python -m backend.examples

Picks one flow per listed label from the processed test split (seed 42) and
writes ``examples/flow_<label>.json`` (a ``/predict`` body) and
``examples/batch.json`` (a ``/predict/batch`` body). The values are real
dataset rows, not invented; the ``Label`` key is kept so you can compare the
API's decision with the ground truth, and the API ignores it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from src.data.preprocess import feature_columns, load_split

LABELS = ("BENIGN", "DDoS", "PortScan", "DoS Hulk", "Bot", "FTP-Patator")


def _value(v: float):
    if math.isnan(v):
        return None
    if math.isinf(v):
        return v  # written as Infinity, which Python's json and the API accept
    return int(v) if float(v).is_integer() else float(v)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/processed/flows.parquet")
    parser.add_argument("--out", default="examples")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    test = load_split(args.data, "test")
    feats = feature_columns(test)
    batch = []
    for label in LABELS:
        rows = test[test["label"] == label]
        if rows.empty:
            continue
        row = rows.sample(n=1, random_state=42).iloc[0]
        flow = {f: _value(float(row[f])) for f in feats} | {"Label": label}
        batch.append(flow)
        name = label.lower().replace(" ", "_").replace("-", "_")
        (out / f"flow_{name}.json").write_text(json.dumps({"features": flow, "source": "example"}, indent=1),
                                              encoding="utf-8")
    (out / "batch.json").write_text(json.dumps({"flows": batch, "source": "example"}, indent=1), encoding="utf-8")
    print(f"wrote {len(batch)} example flows to {out}/")


if __name__ == "__main__":
    main()
