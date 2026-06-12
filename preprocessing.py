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