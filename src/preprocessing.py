"""Leakage-safe feature checks and Train-fitted preprocessing."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def prepare_features(frame: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    """Create a model-only view; preserve original raw data unmodified.

    For zero load only, missing load_position is represented as the categorical
    value NONE. A missing position at nonzero load is a data-quality error.
    """
    features = list(features)
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"Requested feature(s) missing from source: {missing}")
    output = frame.loc[:, features].copy()
    if "load_position" in output:
        absent_position = output["load_position"].isna() | output["load_position"].eq("")
        nonzero_missing = absent_position & frame["load_level_kg"].ne(0)
        if nonzero_missing.any():
            raise ValueError("load_position is missing for a nonzero-load Window")
        output.loc[absent_position, "load_position"] = "NONE"
    numeric = [name for name in features if name != "load_position"]
    if numeric and output[numeric].isna().any().any():
        missing_counts = output[numeric].isna().sum()
        raise ValueError(f"Missing numeric feature values: {missing_counts[missing_counts > 0].to_dict()}")
    return output


def make_preprocessor(features: Iterable[str]) -> ColumnTransformer:
    """Build an unfitted preprocessor. Fit is performed only on Train rows."""
    features = list(features)
    numeric = [name for name in features if name != "load_position"]
    transformers = []
    if numeric:
        transformers.append(("numeric", "passthrough", numeric))
    if "load_position" in features:
        transformers.append(
            ("load_position", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["load_position"])
        )
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
