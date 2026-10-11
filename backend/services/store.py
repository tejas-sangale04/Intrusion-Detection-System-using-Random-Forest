"""In-memory record of recent predictions, which ``/alerts`` and ``/explanations`` read.

Trade-off: a bounded in-memory store needs no database and cannot grow
without limit, but it is emptied when the server restarts and drops the
oldest predictions once full. That suits an offline demo. Alerts can also be
appended to a JSON-lines file (``IDS_ALERT_LOG``) so they survive a restart;
a real deployment would use a database instead.

Nothing here blocks traffic or changes firewall rules. An alert is a record
for an analyst to look at, nothing more.
"""

from __future__ import annotations

import json
import threading
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class PredictionRecord:
    prediction_id: str
    created_at: str
    source: str
    decision: str
    attack_probability: float
    anomaly_score: float
    missing_values: list[str]
    features: dict[str, float] = field(repr=False)  # as received (NaN/inf kept); not returned by the API
    explanation: dict | None = field(default=None, repr=False)

    def summary(self) -> dict:
        out = asdict(self)
        out.pop("features")
        out.pop("explanation")
        return out


class PredictionStore:
    def __init__(self, max_size: int, alert_log: Path | None = None):
        self.max_size = max_size
        self.alert_log = alert_log
        self._records: OrderedDict[str, PredictionRecord] = OrderedDict()
        self._lock = threading.Lock()
        self.decision_counts: Counter[str] = Counter()
        self.evicted = 0
        if alert_log is not None:
            alert_log.parent.mkdir(parents=True, exist_ok=True)

    def add(self, record: PredictionRecord) -> None:
        with self._lock:
            self._records[record.prediction_id] = record
            self.decision_counts[record.decision] += 1
            while len(self._records) > self.max_size:
                self._records.popitem(last=False)
                self.evicted += 1
            if self.alert_log is not None and record.decision != "benign":
                with self.alert_log.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record.summary()) + "\n")

    def get(self, prediction_id: str) -> PredictionRecord | None:
        with self._lock:
            return self._records.get(prediction_id)

    def alerts(self, limit: int, decision: str | None = None) -> list[PredictionRecord]:
        with self._lock:
            records = list(reversed(self._records.values()))
        wanted = {decision} if decision else {"anomaly", "known_attack"}
        return [r for r in records if r.decision in wanted][:limit]

    def __len__(self) -> int:
        return len(self._records)
