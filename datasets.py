"""
datasets.py

Dataset management module for the ML Workbench Dashboard.

Supported:
- Scikit-learn datasets
- Seaborn datasets
- Custom CSV upload
"""

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_diabetes,
    fetch_california_housing
)



# DATASET LOADING FUNCTIONS


def load_iris_dataset():
    """Load Iris dataset."""

    data = load_iris(as_frame=True)

    df = data.frame.copy()

    df["target"] = df["target"].map(
        {
            0: data.target_names[0],
            1: data.target_names[1],
            2: data.target_names[2]
        }
    )

    return df


def load_wine_dataset():
    """Load Wine dataset."""

    data = load_wine(as_frame=True)

    df = data.frame.copy()

    df["target"] = df["target"].map(
        {
            0: data.target_names[0],
            1: data.target_names[1],
            2: data.target_names[2]
        }
    )

    return df


def load_breast_cancer_dataset():
    """Load Breast Cancer dataset."""

    data = load_breast_cancer(as_frame=True)

    df = data.frame.copy()

    df["target"] = df["target"].map(
        {
            0: "malignant",
            1: "benign"
        }
    )

    return df


def load_diabetes_dataset():
    """Load Diabetes Regression dataset."""

    data = load_diabetes(as_frame=True)

    df = data.frame.copy()

    return df


def load_california_housing_dataset():
    """Load California Housing dataset."""

    data = fetch_california_housing(as_frame=True)

    df = data.frame.copy()

    return df


def load_titanic_dataset():
    """Load Titanic dataset from Seaborn."""

    return sns.load_dataset("titanic")


def load_penguins_dataset():
    """Load Penguins dataset."""

    return sns.load_dataset("penguins")


def load_tips_dataset():
    """Load Tips dataset."""

    return sns.load_dataset("tips")


def load_mpg_dataset():
    """Load MPG dataset."""

    return sns.load_dataset("mpg")


def load_mall_customers_dataset():
    """
    Generate a mall customer style clustering dataset.

    This avoids dependency on external CSV.
    """

    np.random.seed(42)

    n = 200

    df = pd.DataFrame({
        "Age": np.random.randint(18, 70, n),
        "Annual_Income": np.random.randint(15, 140, n),
        "Spending_Score": np.random.randint(1, 100, n)
    })

    return df



# DATASET REGISTRY


DATASET_REGISTRY = {
    "Iris": load_iris_dataset,
    "Wine": load_wine_dataset,
    "Breast Cancer": load_breast_cancer_dataset,
    "Diabetes": load_diabetes_dataset,
    "California Housing": load_california_housing_dataset,
    "Titanic": load_titanic_dataset,
    "Penguins": load_penguins_dataset,
    "Tips": load_tips_dataset,
    "MPG": load_mpg_dataset,
    "Mall Customers": load_mall_customers_dataset
}


def get_available_datasets():
    """
    Returns available dataset names.
    """

    return list(DATASET_REGISTRY.keys())


def load_dataset(dataset_name):
    """
    Load dataset by name.
    """

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{dataset_name}' not found."
        )

    return DATASET_REGISTRY[dataset_name]()



# CUSTOM CSV LOADER


def load_uploaded_csv(uploaded_file):
    """
    Load user uploaded CSV.
    """

    try:
        df = pd.read_csv(uploaded_file)

        return df

    except Exception as e:
        raise ValueError(
            f"Error reading CSV: {e}"
        )



# DATASET SUMMARY


def get_dataset_summary(df):
    """
    Returns summary statistics.
    """

    summary = {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": int(df.isna().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum())
    }

    return summary


def get_missing_value_report(df):
    """
    Missing value report.
    """

    report = pd.DataFrame({
        "Column": df.columns,
        "Missing_Count": df.isna().sum().values,
        "Missing_Percentage":
            (
                df.isna().sum().values /
                len(df)
            ) * 100
    })

    return report.sort_values(
        "Missing_Count",
        ascending=False
    )


def get_dtype_report(df):
    """
    Data type report.
    """

    report = pd.DataFrame({
        "Column": df.columns,
        "Data_Type": df.dtypes.astype(str)
    })

    return report



# COLUMN DETECTION


def get_numerical_columns(df):
    """
    Numerical columns.
    """

    return df.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()


def get_categorical_columns(df):
    """
    Categorical columns.
    """

    return df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()



# TARGET SUGGESTION


def suggest_target_columns(df):
    """
    Suggest likely target columns.
    """

    candidates = []

    for col in df.columns:

        unique_values = df[col].nunique()

        if unique_values <= 20:
            candidates.append(col)

    return candidates



# STATISTICS


def get_descriptive_statistics(df):
    """
    Descriptive statistics.
    """

    try:
        return df.describe(include="all").T

    except Exception:
        return pd.DataFrame()



# CLEANING UTILITIES


def remove_duplicates(df):
    """
    Remove duplicate rows.
    """

    return df.drop_duplicates()


def fill_missing_values(df):
    """
    Basic missing value handling.

    Numeric columns (int/float, excluding bool) -> Median
    Everything else (object, category, bool)    -> Mode

    NOTE: The previous implementation routed any non-"object"
    dtype (including pandas "category" and "bool" columns) into
    the `.median()` branch. `.median()` is not defined for
    Categorical dtypes and raises a TypeError, which was the
    source of the "preprocessing" crashes when datasets such as
    Titanic/Penguins (which contain category-dtype columns like
    "class"/"deck") were auto-cleaned.
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

            mode_val = df[col].mode()

            if len(mode_val):
                df[col] = df[col].fillna(mode_val[0])

    return df



# OVERVIEW PACKAGE


def generate_dataset_overview(df):
    """
    Full overview bundle.
    """

    return {
        "summary": get_dataset_summary(df),
        "missing_report": get_missing_value_report(df),
        "dtype_report": get_dtype_report(df),
        "statistics": get_descriptive_statistics(df),
        "numerical_columns": get_numerical_columns(df),
        "categorical_columns": get_categorical_columns(df),
        "target_suggestions": suggest_target_columns(df)
    }