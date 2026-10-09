"""Small hand-built CSVs that reproduce the quirks of CIC-IDS2017.

These fixtures exist only to test the code. Their numbers are invented and
must never be reported as dataset statistics.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

HEADER = (
    " Destination Port, Flow Duration, Fwd Header Length,Flow Bytes/s,"
    " Flow Packets/s, Bwd PSH Flags, Fwd Header Length, Label\n"
)

# Monday: UTF-8, one exact duplicate row inside the file, NaN and Infinity.
MONDAY = HEADER + (
    "80,100,20,1000.5,10,0,20,BENIGN\n"
    "80,100,20,1000.5,10,0,20,BENIGN\n"  # duplicate of the row above
    "443,200,40,NaN,Infinity,0,40,BENIGN\n"
    "53,0,32,Infinity,-Infinity,0,32,BENIGN\n"
)

# Thursday: cp1252 byte 0x96 in a label, a row that also exists in Monday,
# a feature vector that conflicts with Monday's label, and a malformed value.
THURSDAY = (
    HEADER
    + "80,100,20,1000.5,10,0,20,BENIGN\n"  # cross-file duplicate
    + "443,200,40,NaN,Infinity,0,40,DoS Hulk\n"  # same features as a Monday BENIGN row
    + "80,5,20,abc,3,0,20,Web Attack \u2013 Brute Force\n"  # en dash is byte 0x96 in cp1252
    + "22,7,24,15,2,0,24,SSH-Patator\n"
)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "Monday-WorkingHours.pcap_ISCX.csv").write_text(MONDAY, encoding="utf-8")
    with zipfile.ZipFile(raw / "MachineLearningCSV.zip", "w") as zf:
        zf.writestr("MachineLearningCVE/Thursday-WorkingHours.pcap_ISCX.csv", THURSDAY.encode("cp1252"))
        zf.writestr("__MACOSX/MachineLearningCVE/._Thursday-WorkingHours.pcap_ISCX.csv", b"\x00junk")
    return raw
