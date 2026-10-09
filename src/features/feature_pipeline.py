"""Phase 2: the scikit-learn preprocessing pipeline, fitted on training data only.

Run after ``src.data.preprocess``:

    python -m src.features.feature_pipeline

Pipeline steps and the reason for each:

``SchemaGuard``
    Fixes the input contract. It remembers the exact feature names and order
    seen in training, rejects inputs with missing columns, ignores extra ones,
    and reorders columns. The API and replay mode depend on this: a model fed
    columns in a different order fails silently, not loudly.
``ExcludeColumns``
    Removes features a model should not see. Default: ``Destination Port``.
    A port number identifies the targeted *service*, so it can stand in for
    the label (every FTP-Patator flow goes to port 21) and invites shortcut
    learning that does not transfer to other networks. Phase 4 measures how
    much it actually matters by retraining with it.
``NonFiniteToNaN``
    ``Flow Bytes/s`` and ``Flow Packets/s`` contain +inf where duration is zero
    (Phase 1). Infinity is not a usable number for most models, and clipping it
    to a guessed maximum invents data, so it becomes missing and is imputed.
``DropUninformative``
    Learns from the training split which columns are constant or exact copies
    of an earlier column, and drops them. Phase 1 found 8 constant columns and
    5 duplicate pairs on the full data; deciding this on training data only
    keeps the test split from influencing any modelling choice.
``SimpleImputer(median)``
    Fills the few missing rates. The median is robust to the heavy tails seen
    in Phase 1; it is computed from training data only.
``SignedLog1p`` + ``StandardScaler`` (optional, ``scale=True``)
    Only for models that are sensitive to feature scale (Logistic Regression,
    Isolation Forest comparisons). Tree ensembles split on thresholds and are
    unaffected by monotone transforms, so they use the unscaled variant.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_EXCLUDED = ("Destination Port",)


class SchemaGuard(BaseEstimator, TransformerMixin):
    """Enforce the training feature names and order; coerce values to float."""

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SchemaGuard expects a pandas DataFrame with named columns")
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame with named columns")
        expected = list(self.feature_names_in_)
        missing = [c for c in expected if c not in X.columns]
        if missing:
            raise ValueError(f"Missing {len(missing)} required feature(s): {missing}")
        out = X[expected]
        non_numeric = [c for c in expected if not pd.api.types.is_numeric_dtype(out[c])]
        if non_numeric:  # only coerce columns that need it; numeric ones are already fine
            coerced = out[non_numeric].apply(pd.to_numeric, errors="coerce")
            bad = coerced.isna() & out[non_numeric].notna()
            if bad.to_numpy().any():
                raise ValueError(f"Non-numeric values in feature(s): {coerced.columns[bad.any()].tolist()}")
            out = out.assign(**coerced)
        return out.astype("float64")

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class ExcludeColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns: tuple[str, ...] = DEFAULT_EXCLUDED):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        self.excluded_ = [c for c in self.columns if c in X.columns]
        self.feature_names_out_ = np.array([c for c in X.columns if c not in self.excluded_], dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[list(self.feature_names_out_)]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


def _nonfinite_to_nan(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace([np.inf, -np.inf], np.nan)


class DropUninformative(BaseEstimator, TransformerMixin):
    """Drop columns that are constant, or exact copies of an earlier column, in the training data."""

    def fit(self, X: pd.DataFrame, y=None):
        constant = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
        duplicate_of: dict[str, str] = {}
        by_hash: dict[int, list[str]] = {}
        for c in X.columns:
            if c in constant:
                continue
            h = int(pd.util.hash_array(X[c].to_numpy()).sum(dtype=np.uint64))
            for earlier in by_hash.get(h, []):
                if X[c].equals(X[earlier]):
                    duplicate_of[c] = earlier
                    break
            else:
                by_hash.setdefault(h, []).append(c)
        self.constant_ = constant
        self.duplicate_of_ = duplicate_of
        dropped = set(constant) | set(duplicate_of)
        self.feature_names_out_ = np.array([c for c in X.columns if c not in dropped], dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[list(self.feature_names_out_)]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


def signed_log1p(X):
    return np.sign(X) * np.log1p(np.abs(X))


def build_preprocessor(scale: bool = False, exclude: tuple[str, ...] = DEFAULT_EXCLUDED) -> Pipeline:
    steps = [
        ("schema", SchemaGuard()),
        ("exclude", ExcludeColumns(exclude)),
        ("nonfinite", FunctionTransformer(_nonfinite_to_nan, feature_names_out="one-to-one")),
        ("drop", DropUninformative()),
        ("impute", SimpleImputer(strategy="median", keep_empty_features=False)),
    ]
    if scale:
        steps += [
            ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    pipe = Pipeline(steps)
    pipe.set_output(transform="pandas")
    return pipe


def describe(pipe: Pipeline) -> dict:
    """Human-readable record of what the fitted pipeline learned."""
    return {
        "input_features": list(pipe.named_steps["schema"].feature_names_in_),
        "excluded": pipe.named_steps["exclude"].excluded_,
        "dropped_constant": pipe.named_steps["drop"].constant_,
        "dropped_duplicate_of": pipe.named_steps["drop"].duplicate_of_,
        "output_features": list(pipe.get_feature_names_out()),
        "imputation_medians": dict(
            zip(pipe.named_steps["drop"].feature_names_out_, pipe.named_steps["impute"].statistics_.tolist(), strict=True)
        ),
        "scaled": "scale" in pipe.named_steps,
    }


def main(argv: list[str] | None = None) -> None:
    from src.data.preprocess import feature_columns, load_split

    parser = argparse.ArgumentParser(description="Fit the preprocessing pipelines on the training split.")
    parser.add_argument("--data", default="data/processed/flows.parquet")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--metrics-dir", default="reports/metrics/preprocessing")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    train = load_split(args.data, "train")
    X = train[feature_columns(train)]
    logger.info("fitting on %d training rows, %d input features", len(X), X.shape[1])

    models, metrics = Path(args.models_dir), Path(args.metrics_dir)
    models.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    for name, scale in (("preprocessor_tree", False), ("preprocessor_scaled", True)):
        pipe = build_preprocessor(scale=scale).fit(X)
        joblib.dump(pipe, models / f"{name}.joblib")
        info = describe(pipe)
        (metrics / f"{name}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        logger.info("%s: %d -> %d features", name, len(info["input_features"]), len(info["output_features"]))


if __name__ == "__main__":
    main()
