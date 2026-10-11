"""Settings, read from environment variables so nothing is hard-coded.

=========================  ====================  ==============================================
variable                   default               meaning
=========================  ====================  ==============================================
``IDS_MODELS_DIR``         ``models``            folder with the Phase 3 and Phase 5 models
``IDS_REPORTS_DIR``        ``reports/metrics``   folder with the saved offline evaluation
``IDS_ALERT_LOG``          (unset)               optional JSON-lines file every alert is
                                                 appended to; unset keeps alerts in memory only
``IDS_MAX_BATCH``          ``1000``              most flows accepted by ``/predict/batch``
``IDS_STORE_SIZE``         ``10000``             predictions kept in memory for ``/alerts``
                                                 and ``/explanations``; oldest dropped first
``IDS_CORS_ORIGINS``       ``http://localhost:5173``  comma-separated origins allowed to call
                                                 the API from a browser (Vite dev server)
=========================  ====================  ==============================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _origins() -> list[str]:
    raw = os.environ.get("IDS_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    return [o.strip() for o in raw.split(",") if o.strip()]


@dataclass(frozen=True)
class Settings:
    models_dir: Path = field(default_factory=lambda: Path(os.environ.get("IDS_MODELS_DIR", "models")))
    reports_dir: Path = field(default_factory=lambda: Path(os.environ.get("IDS_REPORTS_DIR", "reports/metrics")))
    alert_log: Path | None = field(
        default_factory=lambda: Path(p) if (p := os.environ.get("IDS_ALERT_LOG")) else None)
    max_batch: int = field(default_factory=lambda: int(os.environ.get("IDS_MAX_BATCH", "1000")))
    store_size: int = field(default_factory=lambda: int(os.environ.get("IDS_STORE_SIZE", "10000")))
    cors_origins: list[str] = field(default_factory=_origins)
