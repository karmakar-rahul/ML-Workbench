"""
visualizations.py

Visualization module for ML Workbench Dashboard.

Supports:

Classification
- Confusion Matrix
- ROC Curve
- Precision Recall Curve
- Feature Importance

Regression
- Actual vs Predicted
- Residual Plot
- Error Distribution

Clustering
- Cluster Scatter Plot
- Elbow Curve
- Silhouette Plot

General
- Correlation Heatmap
- PCA Projection
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    silhouette_samples
)

from sklearn.decomposition import PCA



# CONFUSION MATRIX


def plot_confusion_matrix(
    y_true,
    y_pred
):
    """
    Confusion matrix heatmap.
    """

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_title(
        "Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    return fig



# ROC CURVE


def plot_roc_curve(
    y_true,
    y_prob
):
    """
    Binary ROC curve.
    """

    if y_prob is None:
        return None

    try:

        fpr, tpr, _ = roc_curve(
            y_true,
            y_prob[:, 1]
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"AUC={roc_auc:.3f}"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random"
            )
        )

        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )

        return fig

    except Exception:
        return None



# PRECISION RECALL CURVE


def plot_precision_recall_curve(
    y_true,
    y_prob
):
    """
    Precision Recall Curve.
    """

    if y_prob is None:
        return None

    try:

        precision, recall, _ = (
            precision_recall_curve(
                y_true,
                y_prob[:, 1]
            )
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines"
            )
        )

        fig.update_layout(
            title="Precision Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision"
        )

        return fig

    except Exception:
        return None



# FEATURE IMPORTANCE


def plot_feature_importance(
    importance_dict
):
    """
    Feature importance bar chart.
    """

    if not importance_dict:
        return None

    df = pd.DataFrame({

        "Feature":
            list(importance_dict.keys()),

        "Importance":
            list(importance_dict.values())
    })

    df = df.sort_values(
        "Importance",
        ascending=False
    )

    fig = px.bar(
        df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )

    return fig



# ACTUAL VS PREDICTED


def plot_actual_vs_predicted(
    y_true,
    y_pred
):
    """
    Regression comparison.
    """

    fig = px.scatter(

        x=y_true,
        y=y_pred,

        labels={
            "x": "Actual",
            "y": "Predicted"
        },

        title="Actual vs Predicted"
    )

    return fig



# RESIDUAL PLOT


def plot_residuals(
    y_true,
    y_pred
):
    """
    Residual plot.
    """

    residuals = y_true - y_pred

    fig = px.scatter(
        x=y_pred,
        y=residuals,
        labels={
            "x": "Predicted",
            "y": "Residual"
        },
        title="Residual Plot"
    )

    return fig



# ERROR DISTRIBUTION


def plot_error_distribution(
    y_true,
    y_pred
):
    """
    Error histogram.
    """

    errors = y_true - y_pred

    fig = px.histogram(
        errors,
        nbins=30,
        title="Prediction Error Distribution"
    )

    return fig



# CORRELATION HEATMAP


def plot_correlation_heatmap(
    df
):
    """
    Numerical correlation heatmap.
    """

    numeric_df = df.select_dtypes(
        include=np.number
    )

    corr = numeric_df.corr()

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    sns.heatmap(
        corr,
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title(
        "Correlation Heatmap"
    )

    return fig



# PCA PROJECTION


def plot_pca_projection(
    X,
    labels=None
):
    """
    PCA 2D projection.

    Returns None if X does not have at least 2 columns,
    since PCA(n_components=2) would otherwise raise.
    """

    if X.shape[1] < 2:
        return None

    pca = PCA(
        n_components=2
    )

    components = pca.fit_transform(
        X
    )

    df_plot = pd.DataFrame({

        "PC1": components[:, 0],
        "PC2": components[:, 1]
    })

    if labels is not None:
        df_plot["Label"] = labels

        fig = px.scatter(
            df_plot,
            x="PC1",
            y="PC2",
            color="Label",
            title="PCA Projection"
        )

    else:

        fig = px.scatter(
            df_plot,
            x="PC1",
            y="PC2",
            title="PCA Projection"
        )

    return fig



# CLUSTER SCATTER


def plot_clusters(
    X,
    labels
):
    """
    PCA-based cluster visualization.

    Returns None if X does not have at least 2 columns,
    since PCA(n_components=2) would otherwise raise.
    """

    if X.shape[1] < 2:
        return None

    pca = PCA(
        n_components=2
    )

    transformed = pca.fit_transform(
        X
    )

    cluster_df = pd.DataFrame({

        "PC1": transformed[:, 0],
        "PC2": transformed[:, 1],
        "Cluster": labels.astype(str)
    })

    fig = px.scatter(
        cluster_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        title="Cluster Visualization"
    )

    return fig



# ELBOW METHOD


def plot_elbow_curve(
    X,
    max_k=10
):
    """
    Elbow curve for KMeans.

    `max_k` is capped so that KMeans never receives
    n_clusters > n_samples (which would raise).
    """

    from sklearn.cluster import KMeans

    n_samples = X.shape[0]

    max_k = max(
        1,
        min(max_k, n_samples - 1 if n_samples > 1 else 1)
    )

    inertia_values = []

    ks = range(
        1,
        max_k + 1
    )

    for k in ks:

        model = KMeans(
            n_clusters=k,
            random_state=42
        )

        model.fit(X)

        inertia_values.append(
            model.inertia_
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(ks),
            y=inertia_values,
            mode="lines+markers"
        )
    )

    fig.update_layout(
        title="Elbow Method",
        xaxis_title="K",
        yaxis_title="Inertia"
    )

    return fig



# SILHOUETTE PLOT


def plot_silhouette(
    X,
    labels
):
    """
    Silhouette plot.
    """

    unique_clusters = np.unique(
        labels
    )

    if len(unique_clusters) < 2:
        return None

    silhouette_vals = (
        silhouette_samples(
            X,
            labels
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    y_lower = 10

    for cluster in unique_clusters:

        cluster_values = silhouette_vals[
            labels == cluster
        ]

        cluster_values.sort()

        size = len(cluster_values)

        y_upper = y_lower + size

        ax.fill_betweenx(
            np.arange(
                y_lower,
                y_upper
            ),
            0,
            cluster_values
        )

        y_lower = y_upper + 10

    ax.set_title(
        "Silhouette Plot"
    )

    ax.set_xlabel(
        "Silhouette Coefficient"
    )

    ax.set_ylabel(
        "Cluster"
    )

    return fig



# DATASET DISTRIBUTION


def plot_numeric_distribution(
    df,
    column
):
    """
    Distribution of a numeric column.
    """

    fig = px.histogram(
        df,
        x=column,
        nbins=30,
        title=f"{column} Distribution"
    )

    return fig



# CATEGORICAL DISTRIBUTION


def plot_categorical_distribution(
    df,
    column
):
    """
    Category counts.
    """

    counts = (
        df[column]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "Category",
        "Count"
    ]

    fig = px.bar(
        counts,
        x="Category",
        y="Count",
        title=f"{column} Distribution"
    )

    return fig