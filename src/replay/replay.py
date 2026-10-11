"""Phase 9: replay recorded CIC-IDS2017 flows through the API at a set rate.

Start the API (Phase 7) and, optionally, the dashboard (Phase 8), then:

    python -m src.replay.replay                          # held-out test flows, 200 flows/s
    python -m src.replay.replay --files Friday --rate 50 --limit 5000
    python -m src.replay.replay --csv path/to/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

What replay is, and is not
--------------------------
Replay sends recorded flow records to ``/predict/batch`` one batch at a time,
paced to a fixed number of flows per second, so the dashboard fills up the
way it would with a live feed. It is a **simulation**: every flow is tagged
with the source ``replay:<name>``, the dashboard shows a replay banner, and
nothing is captured from or sent to a real network.

* **Timing is synthetic.** The MachineLearningCSV files have no timestamps,
  so the original inter-arrival times cannot be reproduced. Flows are sent
  in file order at the rate you choose.
* **Default input: the held-out test split** (``flows.parquet``, split
  ``test``). Those flows were not used to train or tune either model, so the
  running scores are a fair estimate. ``--csv`` replays a raw file exactly as
  recorded instead; most of its rows were in the training data, so its scores
  only show that the pipeline works, not how well it detects attacks.
* **Ground truth stays here.** The ``Label`` column is never sent to the
  API. The replayer keeps it locally and compares it with each decision, so
  the summary can report recall, false-positive rate and precision.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.io import DataSource, find_label_column, iter_chunks, normalize_columns, read_raw_header, scan_file
from src.data.labels import normalize_labels
from src.data.preprocess import feature_columns

logger = logging.getLogger(__name__)

DECISIONS = ("benign", "anomaly", "known_attack")
MAX_BATCH = 1000  # the API's default IDS_MAX_BATCH

Sender = Callable[[list[dict], str], list[str]]  # (flows, source) -> decisions, same order


# --------------------------------------------------------------------- inputs
def frames_from_split(path: str | Path, split: str, files: str | None, chunk: int = 10_000) -> Iterator[pd.DataFrame]:
    """Yield the chosen split of the processed dataset, in recording order."""
    filters = [("split", "==", split)]
    df = pd.read_parquet(path, filters=filters)
    if files:
        df = df[df["source_file"].str.startswith(files)]
    if df.empty:
        raise ValueError(f"No {split!r} flows match --files {files!r}")
    feats = feature_columns(df)
    for start in range(0, len(df), chunk):
        part = df.iloc[start:start + chunk]
        yield part[feats].assign(label=part["label"].to_numpy(), is_attack=part["is_attack"].to_numpy())


def frames_from_csv(path: str | Path, member: str | None = None, chunk: int = 10_000) -> Iterator[pd.DataFrame]:
    """Yield a raw CIC-IDS2017 CSV (on disk or inside a zip) with the pipeline's header and label handling."""
    source = DataSource(Path(path), member)
    scan = scan_file(source)
    names = normalize_columns(read_raw_header(source, scan.encoding)).names
    label_col = find_label_column(names)
    feats = [n for n in names if n != label_col]
    for part in iter_chunks(source, scan.encoding, names, chunk):
        meta = normalize_labels(part[label_col])
        X = part[feats].apply(pd.to_numeric, errors="coerce")
        yield X.assign(label=meta["label"].to_numpy(), is_attack=meta["is_attack"].to_numpy())


def to_json_flows(X: pd.DataFrame) -> list[dict]:
    """Rows as JSON-safe dicts: NaN and +/-Infinity become null (the API imputes both identically)."""
    values = X.to_numpy(dtype="float64")
    values = np.where(np.isfinite(values), values, np.nan)
    cols = list(X.columns)
    return [{c: (None if math.isnan(v) else v) for c, v in zip(cols, row)} for row in values.tolist()]


# ------------------------------------------------------------------- tallying
@dataclass
class Tally:
    """Decisions per true label, kept locally; the API never sees the labels."""

    counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, labels: list[str], decisions: list[str]) -> None:
        for label, decision in zip(labels, decisions, strict=True):
            row = self.counts.setdefault(label, dict.fromkeys(DECISIONS, 0))
            row[decision] += 1

    def summary(self) -> dict:
        benign = self.counts.get("BENIGN", dict.fromkeys(DECISIONS, 0))
        attacks = {k: v for k, v in self.counts.items() if k != "BENIGN"}
        n_benign = sum(benign.values())
        n_attack = sum(sum(v.values()) for v in attacks.values())

        def rate(num: int, den: int) -> float | None:
            return num / den if den else None

        out = {"flows": n_benign + n_attack, "benign_flows": n_benign, "attack_flows": n_attack}
        # "any alert" counts both anomaly and known_attack, as an analyst would see them.
        for name, flagged in (("known_attack_only", ("known_attack",)), ("any_alert", ("known_attack", "anomaly"))):
            tp = sum(v[d] for v in attacks.values() for d in flagged)
            fp = sum(benign[d] for d in flagged)
            out[name] = {"recall": rate(tp, n_attack), "false_positive_rate": rate(fp, n_benign),
                         "precision": rate(tp, tp + fp), "true_positives": tp, "false_positives": fp}
        out["by_label"] = {k: {**v, "flows": sum(v.values())} for k, v in sorted(self.counts.items())}
        return out


# --------------------------------------------------------------------- replay
@dataclass
class Replayer:
    send: Sender
    rate: float  # flows per second; 0 = as fast as the API answers
    batch: int
    source: str
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    progress_every: float = 5.0  # seconds between progress log lines

    def run(self, frames: Iterator[pd.DataFrame], limit: int | None = None) -> dict:
        if not 1 <= self.batch <= MAX_BATCH:
            raise ValueError(f"--batch must be between 1 and {MAX_BATCH}")
        tally, sent, start = Tally(), 0, self.clock()
        last_log = start
        stopped = "finished"
        try:
            for frame in frames:
                for i in range(0, len(frame), self.batch):
                    part = frame.iloc[i:i + self.batch]
                    if limit is not None:
                        part = part.iloc[:max(0, limit - sent)]
                    if part.empty:
                        break
                    if self.rate > 0:  # wait until this batch's slot on the fixed schedule
                        due = start + sent / self.rate
                        if (wait := due - self.clock()) > 0:
                            self.sleep(wait)
                    flows = to_json_flows(part.drop(columns=["label", "is_attack"]))
                    decisions = self.send(flows, self.source)
                    tally.add(part["label"].tolist(), decisions)
                    sent += len(part)
                    if self.clock() - last_log >= self.progress_every:
                        last_log = self.clock()
                        s = tally.summary()["any_alert"]
                        logger.info("%d flows sent; recall so far %s, false-positive rate %s", sent,
                                    _pct(s["recall"]), _pct(s["false_positive_rate"]))
                if limit is not None and sent >= limit:
                    stopped = "limit reached"
                    break
        except KeyboardInterrupt:
            stopped = "interrupted"
        seconds = self.clock() - start
        return {"source": self.source, "stopped": stopped, "seconds": round(seconds, 2),
                "achieved_rate": round(sent / seconds, 1) if seconds > 0 else None,
                "requested_rate": self.rate, **tally.summary()}


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.2f}%"


def http_sender(api_url: str, timeout: float = 60.0) -> Sender:
    import httpx

    client = httpx.Client(base_url=api_url, timeout=timeout)

    def send(flows: list[dict], source: str) -> list[str]:
        r = client.post("/predict/batch", json={"flows": flows, "source": source})
        if r.status_code != 200:
            raise RuntimeError(f"API returned {r.status_code}: {r.text[:500]}")
        return [p["decision"] for p in r.json()["predictions"]]

    return send


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay recorded flows through the IDS API (simulation).")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--data", default="data/processed/flows.parquet")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--files", help="only source files whose name starts with this, e.g. Friday")
    parser.add_argument("--csv", help="replay this raw CSV (or zip, with --member) instead of the processed split")
    parser.add_argument("--member", help="CSV name inside the --csv zip archive")
    parser.add_argument("--rate", type=float, default=200, help="flows per second; 0 = as fast as possible")
    parser.add_argument("--batch", type=int, default=50, help=f"flows per request, 1-{MAX_BATCH}")
    parser.add_argument("--limit", type=int, help="stop after this many flows")
    parser.add_argument("--report", help="write the final summary as JSON to this path")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is too noisy

    if args.csv:
        name = Path(args.member or args.csv).name
        frames = frames_from_csv(args.csv, args.member)
        logger.warning("Replaying a raw file: most of its rows were used in training, so scores here "
                       "show the pipeline works, not detection quality on unseen traffic.")
    else:
        name = f"{args.split}{'-' + args.files if args.files else ''}"
        frames = frames_from_split(args.data, args.split, args.files)
    source = f"replay:{name}"[:64]

    logger.info("Replaying %s to %s at %s flows/s (simulation; no network capture)", name, args.api,
                args.rate or "max")
    report = Replayer(http_sender(args.api), args.rate, args.batch, source).run(frames, args.limit)
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
