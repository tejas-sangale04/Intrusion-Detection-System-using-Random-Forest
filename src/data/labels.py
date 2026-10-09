"""Label normalisation and the label -> family -> binary mapping.

Phase 1 showed the raw labels exactly as shipped. Three of them contain the
Unicode replacement character (``Web Attack � Brute Force``): the original
dash was lost before the files were published, so it is restored here as a
plain ``-``. Every known label is listed explicitly; an unknown label raises
instead of being silently filed under some family.
"""

from __future__ import annotations

import re

import pandas as pd

BENIGN = "BENIGN"

# Raw characters that stand in for a dash in CIC-IDS2017 releases.
_DASH_LIKE = re.compile(r"\s*[�–—\x96]\s*")
_SPACES = re.compile(r"\s+")

# Attack family groups variants of the same technique, so "hold out one family"
# experiments remove every variant at once (holding out DoS Hulk while
# training on DoS GoldenEye would not test an unseen attack type).
LABEL_TO_FAMILY: dict[str, str] = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "Web Attack - Brute Force": "Web Attack",
    "Web Attack - XSS": "Web Attack",
    "Web Attack - Sql Injection": "Web Attack",
    "Bot": "Bot",
    "Infiltration": "Infiltration",
    "Heartbleed": "Heartbleed",
}


def normalize_label(raw: object) -> str:
    """Return the canonical spelling of one raw label string."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        raise ValueError("Missing label")
    text = _DASH_LIKE.sub(" - ", str(raw).strip())
    text = _SPACES.sub(" ", text)
    if text.upper() == BENIGN:
        return BENIGN
    if text not in LABEL_TO_FAMILY:
        raise ValueError(f"Unknown label {raw!r} (normalised to {text!r}); add it to LABEL_TO_FAMILY")
    return text


def normalize_labels(raw: pd.Series) -> pd.DataFrame:
    """Vectorised over unique values: returns label, family and is_attack columns."""
    mapping = {value: normalize_label(value) for value in pd.unique(raw)}
    label = raw.map(mapping).astype("object")
    family = label.map(LABEL_TO_FAMILY)
    return pd.DataFrame(
        {"label": label, "family": family, "is_attack": (label != BENIGN).astype("int8")},
        index=raw.index,
    )
