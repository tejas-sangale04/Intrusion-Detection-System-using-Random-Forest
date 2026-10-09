"""Read-only access to the raw CIC-IDS2017 CSV files.

Everything here is shared by inspection (Phase 1) and preprocessing (Phase 2),
so both phases see the same file list, encodings and column names.

Design notes
------------
* Files are never modified. CSVs are read either from disk or directly from
  inside ``MachineLearningCSV.zip``, so extracting the archive is optional.
* Encoding is detected per file instead of assumed. Some CIC-IDS2017 releases
  contain a non-UTF-8 byte in the web-attack labels; reading that file as
  UTF-8 would crash, and reading every file as latin-1 would silently garble
  any genuinely UTF-8 file.
* Column names are normalised in exactly one place (``normalize_columns``):
  leading/trailing whitespace is stripped and duplicate names get a ``.1``
  suffix, mirroring pandas, so the rename is visible and reproducible.
"""

from __future__ import annotations

import codecs
import csv
import io
import logging
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import pandas as pd

logger = logging.getLogger(__name__)

SCAN_BLOCK_BYTES = 1 << 20  # 1 MiB blocks for the encoding / line-count scan


@dataclass(frozen=True)
class DataSource:
    """One CSV file, either on disk or as a member of a zip archive."""

    path: Path
    member: str | None = None

    @property
    def name(self) -> str:
        return Path(self.member).name if self.member else self.path.name

    @property
    def location(self) -> str:
        return f"{self.path}!{self.member}" if self.member else str(self.path)

    @contextmanager
    def open_binary(self) -> Iterator[IO[bytes]]:
        if self.member is None:
            with open(self.path, "rb") as fh:
                yield fh
        else:
            with zipfile.ZipFile(self.path) as zf, zf.open(self.member) as fh:
                yield fh

    @contextmanager
    def open_text(self, encoding: str) -> Iterator[IO[str]]:
        with self.open_binary() as fh:
            text = io.TextIOWrapper(fh, encoding=encoding, newline="")
            try:
                yield text
            finally:
                text.detach()  # the binary handle is closed by open_binary


def discover_files(raw_dir: str | Path) -> list[DataSource]:
    """Find every CSV under ``raw_dir``, including CSVs inside zip archives.

    If the same file name exists both extracted and inside an archive, the
    extracted copy wins and the archive member is skipped, so no flow is
    counted twice.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    loose: list[DataSource] = []
    zipped: list[DataSource] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        if suffix == ".csv":
            loose.append(DataSource(path))
        elif suffix == ".zip":
            with zipfile.ZipFile(path) as zf:
                for member in sorted(zf.namelist()):
                    base = Path(member).name
                    if (
                        member.lower().endswith(".csv")
                        and not member.startswith("__MACOSX/")
                        and not base.startswith(".")
                    ):
                        zipped.append(DataSource(path, member))

    loose_names = {s.name for s in loose}
    for source in zipped:
        if source.name in loose_names:
            logger.warning("Skipping %s: an extracted copy is already present", source.location)
    sources = loose + [s for s in zipped if s.name not in loose_names]

    names = [s.name for s in sources]
    duplicated = {n for n in names if names.count(n) > 1}
    if duplicated:
        raise ValueError(f"Several CSV files share a name, refusing to guess: {sorted(duplicated)}")
    if not sources:
        raise FileNotFoundError(
            f"No .csv files (or .zip archives containing them) found under {raw_dir}"
        )
    return sorted(sources, key=lambda s: s.name)


@dataclass
class ScanResult:
    encoding: str
    line_count: int  # physical lines, including the header line
    has_bom: bool = False
    notes: list[str] = field(default_factory=list)


def scan_file(source: DataSource) -> ScanResult:
    """Detect the text encoding and count lines in one streaming pass.

    UTF-8 is tried first. If any byte sequence is invalid UTF-8 the file is
    re-checked as cp1252 (the Windows code page these CSVs were most likely
    written with), and latin-1 is the last resort because it accepts any byte.
    """
    decoder = codecs.getincrementaldecoder("utf-8")()
    utf8_ok = True
    lines = 0
    last_byte = b""
    first_block = True
    has_bom = False
    with source.open_binary() as fh:
        while block := fh.read(SCAN_BLOCK_BYTES):
            if first_block:
                has_bom = block.startswith(codecs.BOM_UTF8)
                first_block = False
            lines += block.count(b"\n")
            last_byte = block[-1:]
            if utf8_ok:
                try:
                    decoder.decode(block)
                except UnicodeDecodeError:
                    utf8_ok = False
    if last_byte and last_byte != b"\n":
        lines += 1  # final line without a trailing newline

    if utf8_ok:
        return ScanResult("utf-8-sig" if has_bom else "utf-8", lines, has_bom)

    notes = ["File is not valid UTF-8."]
    for candidate in ("cp1252", "latin-1"):
        if _decodes_cleanly(source, candidate):
            notes.append(f"Decoded as {candidate}.")
            return ScanResult(candidate, lines, has_bom, notes)
    raise AssertionError("latin-1 decodes any byte sequence")  # pragma: no cover


def _decodes_cleanly(source: DataSource, encoding: str) -> bool:
    decoder = codecs.getincrementaldecoder(encoding)()
    with source.open_binary() as fh:
        try:
            while block := fh.read(SCAN_BLOCK_BYTES):
                decoder.decode(block)
            decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            return False
    return True


def read_raw_header(source: DataSource, encoding: str) -> list[str]:
    with source.open_text(encoding) as fh:
        row = next(csv.reader(fh), None)
    if not row:
        raise ValueError(f"{source.location} has no header row")
    return row


@dataclass
class ColumnNormalization:
    names: list[str]
    stripped: list[tuple[str, str]]  # (raw, normalised) for names that had whitespace
    deduplicated: list[tuple[str, str]]  # (normalised, new unique name)


def normalize_columns(raw_names: list[str]) -> ColumnNormalization:
    """Strip surrounding whitespace and make duplicate names unique.

    CIC-IDS2017 headers look like ``" Destination Port"`` and some names occur
    twice (for example ``Fwd Header Length``). Duplicates are renamed
    ``name.1``, ``name.2`` ... so no column is silently dropped.
    """
    stripped = [(raw, raw.strip()) for raw in raw_names]
    names: list[str] = []
    seen: dict[str, int] = {}
    deduplicated: list[tuple[str, str]] = []
    for _, name in stripped:
        if name in seen:
            seen[name] += 1
            unique = f"{name}.{seen[name]}"
            while unique in seen:
                seen[name] += 1
                unique = f"{name}.{seen[name]}"
            deduplicated.append((name, unique))
            seen[unique] = 0
            names.append(unique)
        else:
            seen[name] = 0
            names.append(name)
    return ColumnNormalization(
        names=names,
        stripped=[(raw, new) for raw, new in stripped if raw != new],
        deduplicated=deduplicated,
    )


def find_label_column(names: list[str]) -> str:
    """Return the target column, matched case-insensitively against ``label``."""
    matches = [n for n in names if n.strip().lower() == "label"]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one 'Label' column, found {matches!r} in {names!r}")
    return matches[0]


def iter_chunks(
    source: DataSource, encoding: str, names: list[str], chunksize: int
) -> Iterator[pd.DataFrame]:
    """Stream the file as DataFrames with the normalised column names.

    Rows whose field count does not match the header are skipped by the
    parser; callers detect them by comparing parsed rows with the line count
    from ``scan_file``.
    """
    with source.open_text(encoding) as fh:
        reader = pd.read_csv(
            fh,
            header=0,
            names=names,
            chunksize=chunksize,
            low_memory=False,
            on_bad_lines="skip",
            skip_blank_lines=True,
        )
        yield from reader
