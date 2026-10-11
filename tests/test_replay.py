from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.replay.replay import Replayer, Tally, frames_from_csv, to_json_flows


def _frame(labels: list[str]) -> pd.DataFrame:
    n = len(labels)
    return pd.DataFrame({
        "Flow Duration": np.arange(n, dtype=float),
        "Flow Bytes/s": [np.inf] + [1.0] * (n - 1),
        "label": labels,
        "is_attack": [int(l != "BENIGN") for l in labels],
    })


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s


def test_flows_are_json_safe_and_labels_never_sent() -> None:
    sent: list[list[dict]] = []

    def send(flows, source):
        sent.append(flows)
        json.dumps(flows, allow_nan=False)  # would raise on NaN/Infinity
        return ["benign"] * len(flows)

    Replayer(send, rate=0, batch=2, source="replay:t").run(iter([_frame(["BENIGN", "DDoS", "BENIGN"])]))
    assert [len(b) for b in sent] == [2, 1]
    assert sent[0][0]["Flow Bytes/s"] is None
    assert all("label" not in f and "is_attack" not in f for b in sent for f in b)


def test_rate_paces_batches_on_a_fixed_schedule() -> None:
    clock = FakeClock()
    r = Replayer(lambda f, s: ["benign"] * len(f), rate=10, batch=5, source="replay:t",
                 clock=clock.clock, sleep=clock.sleep)
    report = r.run(iter([_frame(["BENIGN"] * 20)]))
    # batches due at 0, 0.5, 1.0, 1.5 s: three waits of 0.5 s
    assert clock.sleeps == pytest.approx([0.5, 0.5, 0.5])
    assert report["flows"] == 20 and report["stopped"] == "finished"


def test_limit_stops_mid_batch() -> None:
    report = Replayer(lambda f, s: ["benign"] * len(f), rate=0, batch=4, source="replay:t").run(
        iter([_frame(["BENIGN"] * 10), _frame(["BENIGN"] * 10)]), limit=6)
    assert report["flows"] == 6 and report["stopped"] == "limit reached"


def test_tally_reports_recall_fpr_precision() -> None:
    t = Tally()
    t.add(["BENIGN", "BENIGN", "BENIGN", "BENIGN", "DDoS", "DDoS", "Bot"],
          ["benign", "benign", "anomaly", "benign", "known_attack", "anomaly", "benign"])
    s = t.summary()
    assert s["known_attack_only"]["recall"] == pytest.approx(1 / 3)
    assert s["any_alert"]["recall"] == pytest.approx(2 / 3)
    assert s["any_alert"]["false_positive_rate"] == pytest.approx(1 / 4)
    assert s["any_alert"]["precision"] == pytest.approx(2 / 3)
    assert s["by_label"]["Bot"] == {"benign": 1, "anomaly": 0, "known_attack": 0, "flows": 1}


def test_raw_csv_uses_pipeline_header_and_label_handling(raw_dir) -> None:
    zip_path = raw_dir / "MachineLearningCSV.zip"
    frames = list(frames_from_csv(zip_path, "MachineLearningCVE/Thursday-WorkingHours.pcap_ISCX.csv"))
    df = pd.concat(frames)
    assert "Fwd Header Length.1" in df.columns  # duplicate header renamed, as in Phase 1
    assert "Web Attack - Brute Force" in set(df["label"])  # cp1252 dash normalised
    assert df["is_attack"].sum() == 3
    assert np.isnan(df.loc[df["label"] == "Web Attack - Brute Force", "Flow Bytes/s"]).all()  # "abc" -> missing
    assert to_json_flows(df.drop(columns=["label", "is_attack"]).iloc[[1]])[0]["Flow Packets/s"] is None
