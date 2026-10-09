from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.feature_pipeline import build_preprocessor, describe


@pytest.fixture
def train() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 50
    return pd.DataFrame(
        {
            "Destination Port": rng.integers(1, 65535, n).astype(float),
            "Flow Duration": rng.integers(0, 10_000, n).astype(float),
            "Flow Bytes/s": np.r_[np.inf, np.nan, rng.random(n - 2) * 1e6],
            "Total Fwd Packets": rng.integers(1, 50, n).astype(float),
            "Subflow Fwd Packets": 0.0,  # filled below as an exact copy
            "Bwd PSH Flags": 0.0,  # constant
            "Init_Win_bytes_forward": np.r_[-1.0, rng.integers(0, 65535, n - 1)],
        }
    ).assign(**{"Subflow Fwd Packets": lambda d: d["Total Fwd Packets"]})


def test_pipeline_learns_drops_from_training_data(train: pd.DataFrame) -> None:
    pipe = build_preprocessor().fit(train)
    info = describe(pipe)
    assert info["excluded"] == ["Destination Port"]
    assert info["dropped_constant"] == ["Bwd PSH Flags"]
    assert info["dropped_duplicate_of"] == {"Subflow Fwd Packets": "Total Fwd Packets"}
    assert info["output_features"] == ["Flow Duration", "Flow Bytes/s", "Total Fwd Packets", "Init_Win_bytes_forward"]


def test_infinite_and_missing_values_are_imputed_with_training_median(train: pd.DataFrame) -> None:
    pipe = build_preprocessor().fit(train)
    median = train["Flow Bytes/s"].replace(np.inf, np.nan).median()
    out = pipe.transform(train)
    assert np.isfinite(out.to_numpy()).all()
    assert out["Flow Bytes/s"].iloc[0] == pytest.approx(median)
    assert out["Flow Bytes/s"].iloc[1] == pytest.approx(median)
    # The -1 sentinel is a real value in the data, not missing: kept as is.
    assert out["Init_Win_bytes_forward"].iloc[0] == -1


def test_column_order_does_not_matter_and_extras_are_ignored(train: pd.DataFrame) -> None:
    pipe = build_preprocessor().fit(train)
    shuffled = train[train.columns[::-1]].assign(extra_column=1.0)
    pd.testing.assert_frame_equal(pipe.transform(shuffled), pipe.transform(train))


def test_missing_feature_is_rejected(train: pd.DataFrame) -> None:
    pipe = build_preprocessor().fit(train)
    with pytest.raises(ValueError, match="Missing 1 required feature"):
        pipe.transform(train.drop(columns="Flow Duration"))


def test_malformed_value_is_rejected(train: pd.DataFrame) -> None:
    pipe = build_preprocessor().fit(train)
    bad = train.astype(object)
    bad.loc[3, "Flow Duration"] = "abc"
    with pytest.raises(ValueError, match="Non-numeric"):
        pipe.transform(bad)


def test_numpy_input_is_rejected(train: pd.DataFrame) -> None:
    with pytest.raises(TypeError):
        build_preprocessor().fit(train.to_numpy())


def test_scaled_variant_is_standardised_on_training_data(train: pd.DataFrame) -> None:
    out = build_preprocessor(scale=True).fit_transform(train)
    assert np.allclose(out.mean(), 0, atol=1e-9)
    assert np.allclose(out.std(ddof=0), 1, atol=1e-9)


def test_test_data_does_not_change_what_was_learned(train: pd.DataFrame) -> None:
    pipe = build_preprocessor().fit(train)
    before = describe(pipe)
    test = train.assign(**{"Bwd PSH Flags": 5.0, "Flow Bytes/s": 1e12})
    pipe.transform(test)
    assert describe(pipe) == before
