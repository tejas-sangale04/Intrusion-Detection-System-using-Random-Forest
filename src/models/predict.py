"""Score flow records with the saved binary model.

    python -m src.models.predict --input flows.csv --output predictions.csv

The input must contain every feature the model was trained on (names as in
the CIC-IDS2017 CSVs; surrounding whitespace is ignored, extra columns such
as Label are ignored). Missing or non-numeric features are rejected by the
pipeline's schema guard rather than guessed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def load_model(models_dir: str | Path = "models"):
    models_dir = Path(models_dir)
    pipe = joblib.load(models_dir / "binary_best.joblib")
    card = json.loads((models_dir / "binary_best.json").read_text(encoding="utf-8"))
    return pipe, card


def predict_frame(pipe, frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    frame = frame.rename(columns=lambda c: str(c).strip())
    score = pipe.predict_proba(frame)[:, 1]
    return pd.DataFrame({"attack_probability": score, "is_attack": (score >= threshold).astype(int)}, index=frame.index)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--threshold", type=float, help="defaults to the threshold in the model card")
    args = parser.parse_args(argv)

    pipe, card = load_model(args.models_dir)
    frame = pd.read_csv(args.input, low_memory=False)
    out = predict_frame(pipe, frame, args.threshold if args.threshold is not None else card["threshold"])
    out.to_csv(args.output, index=False)
    print(f"{len(out):,} flows scored with {card['model']}; {int(out['is_attack'].sum()):,} flagged as attack")


if __name__ == "__main__":
    main()
