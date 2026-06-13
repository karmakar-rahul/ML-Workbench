"""
preprocessing.py

Preprocessing module for ML Workbench Dashboard.

Features:
- Missing value handling
- Duplicate removal
- Categorical encoding
- Target encoding
- Feature scaling
- Outlier removal
- Problem type detection
- Data quality reporting
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)



# MISSING VALUE HANDLING


def fill_missing_values(df):
    """
    Fill missing values.

    Numeric columns (excluding bool) -> Median
    Everything else (object, category, bool) -> Mode
    """

    df = df.copy()

    for col in df.columns:

        if df[col].isna().sum() == 0:
            continue

        is_numeric = (
            pd.api.types.is_numeric_dtype(df[col])
            and not pd.api.types.is_bool_dtype(df[col])
        )

        if is_numeric:

            df[col] = df[col].fillna(
                df[col].median()
            )

        else:

            mode_values = df[col].mode()

            if len(mode_values) > 0:

                df[col] = df[col].fillna(
                    mode_values[0]
                )

    return df



# REMOVE DUPLICATES


def remove_duplicates(df):
    """
    Remove duplicate rows.
    """

    return df.drop_duplicates()



# ENCODE CATEGORICAL FEATURES


def encode_categorical_features(X):
    """
    Label encode categorical features.

    Returns:
        X_encoded
        encoders
    """

    X = X.copy()

    encoders = {}

    categorical_columns = X.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns

    for col in categorical_columns:

        encoder = LabelEncoder()

        X[col] = encoder.fit_transform(
            X[col].astype(str)
        )

        encoders[col] = encoder

    return X, encoders



# ENCODE TARGET


def encode_target(y):
    """
    Encode target if categorical.

    Returns:
        encoded_target
        target_encoder
    """

    if (
        str(y.dtype) == "object"
        or str(y.dtype) == "category"
    ):

        encoder = LabelEncoder()

        y_encoded = encoder.fit_transform(
            y.astype(str)
        )

        return y_encoded, encoder

    return y, None



# FEATURE SCALING


def scale_features(
    X,
    method="standard"
):
    """
    Scale features.

    Supported methods:
    - standard
    - minmax
    - robust
    """

    X = X.copy()

    if method == "standard":

        scaler = StandardScaler()

    elif method == "minmax":

        scaler = MinMaxScaler()

    elif method == "robust":

        scaler = RobustScaler()

    else:

        raise ValueError(
            f"Unsupported scaling method: {method}"
        )

    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    return X_scaled, scaler



# OUTLIER REMOVAL USING IQR


def remove_outliers_iqr(
    df,
    multiplier=1.5
):
    """
    Remove outliers using IQR method.
    """

    df = df.copy()

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    mask = pd.Series(
        True,
        index=df.index
    )

    for col in numeric_columns:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower_bound = (
            q1 - multiplier * iqr
        )

        upper_bound = (
            q3 + multiplier * iqr
        )

        mask &= (
            (df[col] >= lower_bound)
            &
            (df[col] <= upper_bound)
        )

    return df[mask]



# PROBLEM TYPE DETECTION


def detect_problem_type(y):
    """
    Detect Classification vs Regression.

    Classification:
        - Object dtype
        - Category dtype
        - <= 20 unique values

    Regression:
        - Numeric target
        - > 20 unique values
    """

    y_series = pd.Series(y)

    if (
        str(y_series.dtype) == "object"
        or str(y_series.dtype) == "category"
    ):

        return "Classification"

    unique_count = y_series.nunique()

    if unique_count <= 20:

        return "Classification"

    return "Regression"



# CLASS IMBALANCE DETECTION


def detect_class_imbalance(
    y,
    threshold=0.20
):
    """
    Detect class imbalance.

    Returns:
        is_imbalanced
        distribution_df
    """

    distribution = (
        pd.Series(y)
        .value_counts(normalize=True)
    )

    imbalance_detected = (
        distribution.min()
        < threshold
    )

    distribution_df = pd.DataFrame({

        "Class": distribution.index,
        "Percentage": distribution.values * 100

    })

    return (
        imbalance_detected,
        distribution_df
    )



# DATA LEAKAGE DETECTION


def detect_leakage_columns(
    df,
    target_column,
    feature_columns,
    purity_threshold=0.98,
    max_unique_ratio=0.9
):
    """
    Flag feature columns that (almost) perfectly determine the
    target column.

    This catches classic "target leakage" situations - for
    example, the Titanic dataset's "alive" column is just an
    inverted re-encoding of "survived". If such a column is left
    in the feature set, models can reach ~100% accuracy, which
    looks like a bug ("unreal" accuracy / overfitting) but is
    really information leakage rather than a modeling issue.

    For each candidate feature column, rows are grouped by the
    feature's value and the proportion of rows in each group that
    share the group's majority target value is computed
    ("purity"). The overall (size-weighted) purity is compared
    against `purity_threshold`.

    Columns that are (almost) unique per row (e.g. IDs, names) are
    skipped via `max_unique_ratio`, since a column being unique per
    row trivially gives "perfect" purity without being a leakage
    indicator in the same sense.

    Returns:
        dict mapping column name -> weighted purity score (0-1)
        for every column considered suspicious.
    """

    suspicious = {}

    if target_column not in df.columns:
        return suspicious

    y = df[target_column]

    n_rows = len(df)

    if n_rows == 0:
        return suspicious

    for col in feature_columns:

        if col not in df.columns or col == target_column:
            continue

        x = df[col]

        n_unique = x.nunique(dropna=True)

        if n_unique <= 1:
            continue

        if n_unique >= max_unique_ratio * n_rows:
            continue

        try:

            paired = pd.DataFrame({
                "feature": x,
                "target": y
            }).dropna()

            if len(paired) == 0:
                continue

            grouped = paired.groupby("feature")["target"]

            group_purity = grouped.apply(
                lambda s: s.value_counts(normalize=True).iloc[0]
            )

            group_weight = grouped.size() / len(paired)

            weighted_purity = float(
                (group_purity * group_weight).sum()
            )

        except Exception:
            continue

        if weighted_purity >= purity_threshold:
            suspicious[col] = weighted_purity

    return suspicious



# FEATURE PREPROCESSING PIPELINE


def preprocess_features(
    X,
    fill_missing=True,
    scale=True,
    scaling_method="standard"
):
    """
    Complete feature preprocessing.
    """

    X = X.copy()

    if fill_missing:

        X = fill_missing_values(X)

    X, encoders = (
        encode_categorical_features(X)
    )

    scaler = None

    if scale:

        X, scaler = scale_features(
            X,
            method=scaling_method
        )

    return (
        X,
        encoders,
        scaler
    )



# TARGET PREPROCESSING PIPELINE


def preprocess_target(y):
    """
    Complete target preprocessing.
    """

    y_encoded, encoder = (
        encode_target(y)
    )

    return (
        y_encoded,
        encoder
    )



# DATA QUALITY REPORT


def generate_data_quality_report(df):
    """
    Generate quick dataset quality report.
    """

    report = {

        "rows":
            df.shape[0],

        "columns":
            df.shape[1],

        "missing_values":
            int(
                df.isna()
                .sum()
                .sum()
            ),

        "duplicate_rows":
            int(
                df.duplicated()
                .sum()
            ),

        "numerical_columns":
            len(
                df.select_dtypes(
                    include=np.number
                ).columns
            ),

        "categorical_columns":
            len(
                df.select_dtypes(
                    exclude=np.number
                ).columns
            )
    }

    return report
