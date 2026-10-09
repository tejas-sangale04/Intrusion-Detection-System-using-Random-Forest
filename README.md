# Explainable Hybrid Network Intrusion Detection System

A research-oriented network intrusion detection system (NIDS) built on the
CIC-IDS2017 flow features: a supervised Random Forest, an unsupervised
Isolation Forest, SHAP explanations, a FastAPI backend and a React dashboard.

This is an academic portfolio project. It does not claim production-grade
security or guaranteed detection of unseen ("zero-day") attacks.

## Status

| Phase | What | State |
|---|---|---|
| 1 | Dataset discovery, inspection and EDA | **Done** (all 8 files, see below) |
| 2 | Cleaning and leakage-safe preprocessing | **Done** (see below) |
| 3 | Baseline models (LR, DT, RF, HistGradientBoosting) | Planned |
| 4 | Leakage control and held-out attack-family evaluation | Planned |
| 5 | Isolation Forest and hybrid decision policy | Planned |
| 6 | SHAP explanations | Planned |
| 7 | FastAPI backend | Planned |
| 8 | React dashboard | Planned |
| 9 | Dataset replay, then optional live flows in an authorised lab | Planned |
| 10 | Optional reinforcement-learning response simulation | Planned |

Every dataset number or model result in this README was produced by the code
in this repository.

## Project layout (current)

```
data/raw/           original CSVs or MachineLearningCSV.zip (never modified, not in Git)
data/processed/     generated samples and, later, cleaned data (not in Git)
notebooks/          01_dataset_inspection, 02_exploratory_data_analysis
src/data/io.py      file discovery, encoding detection, header normalisation, chunked reading
src/data/inspect_dataset.py   streaming statistics, duplicates, samples
src/data/eda.py     figures, correlation analysis, generated report
src/data/labels.py  label normalisation, label -> family -> binary mapping
src/data/preprocess.py        de-duplication, float32 parquet, stratified split
src/features/feature_pipeline.py  scikit-learn preprocessing fitted on train only
reports/            dataset_inspection.md, figures/inspection/, metrics/inspection/
tests/              pytest suite using small hand-built CSVs
```

Folders for later phases (`src/features`, `src/models`, `backend`, `frontend`
...) are created when those phases are built, so the tree only shows what exists.

## Setup (Windows, VS Code)

```powershell
cd ids
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest
```

On macOS/Linux use `python3 -m venv .venv` and `source .venv/bin/activate`.
In VS Code, pick `.venv` with *Python: Select Interpreter*.

## Dataset

1. Download `MachineLearningCSV.zip` from the Canadian Institute for
   Cybersecurity CIC-IDS2017 page (https://www.unb.ca/cic/datasets/ids-2017.html).
2. Put either the zip itself or its extracted `.csv` files anywhere under
   `data/raw/`. The code reads CSVs straight from the zip, so extracting is
   optional. If both exist, the extracted copy is used and the zip member is
   skipped so no row is counted twice.

## Phase 1: inspection and EDA

```powershell
python -m src.data.inspect_dataset --raw-dir data/raw
python -m src.data.eda
```

The first command streams every file in 200,000-row chunks, so the full
dataset never sits in memory at once. The second makes the figures and writes
`reports/dataset_inspection.md`. On a synthetic file of the same width the
scan ran at roughly 25,000 rows per second with about 1 GB peak memory; real
timings will be recorded after the first real run.

Outputs:

| File | Contents |
|---|---|
| `reports/dataset_inspection.md` | Generated report; every number is measured |
| `reports/metrics/inspection/summary.json` | Headline facts (files, rows, labels, constant / identical / identifier columns, duplicates) |
| `reports/metrics/inspection/files.csv` | Per file: encoding, rows, unparsed lines, header whitespace, duplicate column names |
| `reports/metrics/inspection/label_distribution.csv` | Label x file counts |
| `reports/metrics/inspection/column_stats.csv` | Per feature: NaN, +inf, -inf, unparseable, mean, std, min, max, zeros, negatives |
| `reports/metrics/inspection/nonfinite_by_file.csv` | NaN / inf counts per file and column |
| `reports/metrics/inspection/duplicates_by_label.csv`, `duplicates_by_file.csv` | Redundant duplicate copies |
| `reports/metrics/inspection/correlated_pairs.csv` | Feature pairs with \|r\| >= 0.95 |
| `reports/figures/inspection/*.png` | Class distribution, labels by file, feature distributions, correlation heatmap |

### Design decisions in Phase 1

* **Read-only and reproducible.** Raw files are opened read-only (a test
  checks they are byte-for-byte unchanged). Samples use a fixed seed.
* **Encoding is detected, not assumed.** Each file is checked as UTF-8 first,
  then cp1252, then latin-1. Some CIC-IDS2017 copies have a non-UTF-8 dash in
  the web-attack labels; the label is kept exactly as decoded and flagged as
  non-ASCII so Phase 2 can normalise it deliberately.
* **Headers are normalised in one place.** Surrounding whitespace is stripped
  and repeated names get a `.1` suffix (CIC-IDS2017 is known to repeat
  `Fwd Header Length`), so nothing is silently dropped. The label column is
  located by name and the run stops if it is missing or ambiguous.
* **Files must share a schema.** A file with different columns stops the run
  instead of being merged with misaligned features.
* **Malformed input is counted, not hidden.** Lines the CSV parser cannot read,
  completely empty rows, NaN, +inf, -inf and non-numeric cells are each
  counted separately (`nan` counts every missing value after parsing,
  including unparseable cells, which are also counted on their own).
* **Duplicates are measured because they cause leakage.** If identical flows
  land in both train and test, test scores partly measure memorisation. The
  scan reports exact duplicates within and across files, per label, and
  feature vectors that appear with different labels (which no model can
  classify perfectly). Rows are compared by 64-bit hashes; with a few million
  rows a false match is vanishingly unlikely (about 1 in 10 million).
* **Statistics are exact; plots use samples.** Counts, means and standard
  deviations use every row (mean and variance with a numerically stable
  streaming update). Plots use two exact random samples: a uniform one that
  keeps the real class mix (for correlations), and a stratified one with up
  to 5,000 rows per label so rare attacks such as Heartbleed are visible (for
  per-class plots). The stratified sample must never be used to estimate class
  proportions.
* **Identifier-like columns are flagged, not dropped yet.** Columns such as
  `Destination Port` can act as shortcuts to the label. Phase 2 decides on
  them with evidence and documents the trade-off.
* **No seaborn.** Matplotlib covers every Phase 1 plot, so one fewer dependency.

### Phase 1 findings (full MachineLearningCSV, 8 files)

Measured by the commands above; full detail in `reports/dataset_inspection.md`.

* **2,830,743 rows, 78 features + `Label`, 15 labels.** 80.30% BENIGN. The
  rarest classes are tiny: Infiltration 36, Web Attack Sql Injection 21,
  Heartbleed 11 rows. Accuracy is therefore meaningless as a headline metric:
  predicting BENIGN for everything already scores 80.3%.
* **Every attack label occurs in exactly one file (one day).** BENIGN appears
  in all eight. A split by day therefore means the test day's attacks were
  never seen in training. That is the held-out-attack experiment, not a
  normal supervised test, and the evaluation design in Phase 4 has to treat
  it that way.
* **All files parse cleanly as UTF-8 with identical headers** (whitespace in
  65 of 79 names, `Fwd Header Length` duplicated). The three web-attack labels
  contain the Unicode replacement character (`Web Attack � Brute Force`), so
  the original dash was already lost before distribution; Phase 2 maps them
  to `Web Attack - ...`.
* **Non-finite values are confined to two rate columns:** `Flow Bytes/s`
  (1,358 NaN, 1,509 +inf) and `Flow Packets/s` (2,867 +inf). These are rates
  divided by a zero duration.
* **10.89% of rows (308,381) are redundant exact duplicates**, 256,479 within
  a file and 51,902 across files. They are concentrated in attacks:
  45.4% of SSH-Patator, 42.9% of PortScan, 25.3% of FTP-Patator and 25.2% of
  DoS Hulk rows are repeats. A random row split would put copies of the same
  flow on both sides and inflate test scores, so duplicates are removed
  before splitting.
* **698 feature vectors appear with two labels** (7,020 rows), mostly
  BENIGN vs PortScan (564) and BENIGN vs DoS Hulk (130). These flows are
  indistinguishable from the features alone, so no model can be perfect on them.
* **Unusable columns:** 8 columns are constant over the whole dataset (all
  `Bulk` rate fields plus `Bwd PSH Flags`, `Bwd URG Flags`), and 5 pairs of columns
  are exact copies (`Total Fwd Packets` = `Subflow Fwd Packets`,
  `Total Backward Packets` = `Subflow Bwd Packets`, `Fwd PSH Flags` =
  `SYN Flag Count`, `Fwd URG Flags` = `CWE Flag Count`, and the duplicated
  `Fwd Header Length`). Some identities found in a single file (for example
  `Subflow Fwd Bytes`) do not hold globally, which is why they are measured on
  all rows rather than a sample.
* **`Destination Port`** is the only identifier-like column; there are no
  IPs, timestamps or flow IDs in this release of the CSVs.

## Phase 2: cleaning and preprocessing

```powershell
python -m src.data.preprocess            # -> data/processed/flows.parquet (+ split column)
python -m src.features.feature_pipeline  # -> models/preprocessor_tree.joblib, preprocessor_scaled.joblib
```

The work is split between two places on purpose:

* **Fixed, data-independent steps** run once in `src/data/preprocess.py`:
  label normalisation, exact-duplicate removal, float32 storage and the split.
  None of them looks at feature distributions, so none can leak test
  information into a model.
* **Learned steps** live in a scikit-learn `Pipeline` (`src/features/feature_pipeline.py`)
  that is fitted on the training split only and saved with joblib: schema
  enforcement, excluding `Destination Port`, turning +/-inf into missing,
  dropping columns that are constant or duplicated *in the training data*,
  median imputation, and (for scale-sensitive models only) a signed log
  transform plus standardisation.

| Decision | Why | Trade-off |
|---|---|---|
| Remove exact duplicates (same features and label) before splitting | Otherwise copies of one flow land in train and test and inflate scores | Changes class frequencies (PortScan loses 43%), so per-class counts differ from the raw dataset |
| Keep same-features/different-label rows | The ambiguity is real; hiding it would flatter the model | Caps achievable precision on BENIGN vs PortScan / DoS Hulk |
| Restore the lost dash in web-attack labels; list every label explicitly | Readable labels; an unknown label must fail loudly | New label spellings need one line added |
| Group labels into families (DoS, Brute Force, Web Attack, ...) | Holding out "an unseen attack" must remove all variants of it | Family boundaries are a judgement call, documented in `labels.py` |
| Exclude `Destination Port` by default | Measured on the cleaned data, the port nearly identifies some labels: FTP-Patator goes to port 21 and SSH-Patator to 22 in over 99.9% of flows, and Heartbleed and Infiltration always go to 444. That is a property of this lab setup, not of the attacks, so it would not transfer to other networks | May cost some accuracy; Phase 4 retrains with it to measure how much |
| inf -> NaN -> training median | Infinity comes from dividing by a zero duration; clipping would invent a value | Imputed rows look like typical flows on those two columns |
| Drop constant/duplicate columns learned on train | Removes 15 useless inputs without letting test data decide anything | Recomputed if the training data changes |
| Float32 storage | Halves memory to fit 8 GB laptops | ~7 significant digits; two pairs that differ only beyond that (`Avg Fwd/Bwd Segment Size` vs packet-length means) become identical and are dropped as duplicates |
| Stratified 70/15/15 split by fine label | Every class, even Heartbleed, appears in train, validation and test | Random split = same days in train and test; it measures in-distribution performance only. Day- and family-held-out splits come in Phase 4 |

Measured results (full dataset):

* 2,830,743 rows in, **2,522,362 after removing 308,381 exact duplicates**
  (BENIGN 176,613, PortScan 68,111, DoS Hulk 58,224, SSH-Patator 2,678,
  FTP-Patator 2,005, others under 500).
* Split: **1,765,653 train / 378,354 validation / 378,355 test**. The smallest
  classes are very small in evaluation: Heartbleed 8/2/1, Sql Injection
  15/3/3, Infiltration 25/6/5. Per-class metrics for these are anecdotes, not
  estimates, and will be reported with their counts.
* The fitted pipeline maps **78 input features to 62 model features**:
  `Destination Port` excluded, 8 constant columns dropped, and 7 duplicates
  dropped (`SYN Flag Count`, `CWE Flag Count`, `Avg Fwd Segment Size`,
  `Avg Bwd Segment Size`, `Fwd Header Length.1`, `Subflow Fwd Packets`,
  `Subflow Bwd Packets`). Full record: `reports/metrics/preprocessing/`.

## Limitations (to be expanded with real findings)

* CIC-IDS2017 was recorded in 2017 in a lab network with scripted attacks; it
  is old, heavily imbalanced, and differs from real enterprise traffic.
  Results on it do not imply performance on modern production networks.
* Each attack type was recorded on a single day (confirmed in Phase 1), so
  attack-family and day effects are confounded. Phase 4 examines this.

## Ethical use

Any traffic generation, replay against live hosts or response action must stay
inside an authorised lab environment. The system logs alerts and simulates
responses; it does not block hosts or change firewall rules.
