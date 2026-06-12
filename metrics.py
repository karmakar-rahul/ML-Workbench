"""
metrics.py

Evaluation metrics module for ML Workbench Dashboard.

Supports:
- Classification Metrics
- Regression Metrics
- Clustering Metrics


"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,

    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,

    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)



# CONFUSION MATRIX UTILITIES


def compute_confusion_matrix(
    y_true,
    y_pred
):
    """
    Returns confusion matrix.
    """

    return confusion_matrix(
        y_true,
        y_pred
    )


def compute_binary_rates(
    y_true,
    y_pred
):
    """
    Computes:

    TP
    TN
    FP
    FN

    TPR
    FPR
    Specificity
    Sensitivity

    Binary classification only.
    """

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    if cm.shape != (2, 2):
        return None

    tn, fp, fn, tp = cm.ravel()

    tpr = tp / (tp + fn) if (tp + fn) else 0

    fpr = fp / (fp + tn) if (fp + tn) else 0

    specificity = tn / (tn + fp) if (tn + fp) else 0

    sensitivity = tpr

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TPR": tpr,
        "FPR": fpr,
        "Specificity": specificity,
        "Sensitivity": sensitivity
    }



# CLASSIFICATION METRICS


def classification_metrics(
    y_true,
    y_pred,
    y_prob=None
):
    """
    Compute classification metrics.
    """

    metrics = {}

    metrics["Accuracy"] = accuracy_score(
        y_true,
        y_pred
    )

    metrics["Precision"] = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    metrics["Recall"] = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    metrics["F1 Score"] = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # Binary Metrics

    binary_metrics = compute_binary_rates(
        y_true,
        y_pred
    )

    if binary_metrics:
        metrics.update(binary_metrics)

    # ROC AUC

    try:

        if y_prob is not None:

            if y_prob.shape[1] == 2:

                metrics["ROC AUC"] = roc_auc_score(
                    y_true,
                    y_prob[:, 1]
                )

            else:

                metrics["ROC AUC"] = roc_auc_score(
                    y_true,
                    y_prob,
                    multi_class="ovr"
                )

    except Exception:
        metrics["ROC AUC"] = np.nan

    return metrics



# CLASSIFICATION REPORT


def classification_report_dataframe(
    y_true,
    y_pred
):
    """
    Returns sklearn classification report
    as dataframe.
    """

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    return pd.DataFrame(
        report
    ).transpose()



# CLASSIFICATION METRICS DF


def classification_metrics_dataframe(
    y_true,
    y_pred,
    y_prob=None
):
    """
    Metrics dataframe for export.
    """

    metrics = classification_metrics(
        y_true,
        y_pred,
        y_prob
    )

    return pd.DataFrame(
        list(metrics.items()),
        columns=["Metric", "Value"]
    )



# REGRESSION METRICS


def regression_metrics(
    y_true,
    y_pred,
    n_features=None
):
    """
    Compute regression metrics.
    """

    metrics = {}

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_true,
        y_pred
    )

    mape = mean_absolute_percentage_error(
        y_true,
        y_pred
    )

    metrics["MAE"] = mae
    metrics["MSE"] = mse
    metrics["RMSE"] = rmse
    metrics["R2"] = r2
    metrics["MAPE"] = mape

    # Adjusted R²

    try:

        if n_features:

            n = len(y_true)

            adjusted_r2 = 1 - (
                ((1 - r2) * (n - 1))
                /
                (n - n_features - 1)
            )

            metrics["Adjusted R2"] = adjusted_r2

    except Exception:
        pass

    return metrics



# REGRESSION METRICS DF


def regression_metrics_dataframe(
    y_true,
    y_pred,
    n_features=None
):
    """
    Regression metrics dataframe.
    """

    metrics = regression_metrics(
        y_true,
        y_pred,
        n_features
    )

    return pd.DataFrame(
        list(metrics.items()),
        columns=["Metric", "Value"]
    )



# CLUSTERING METRICS


def clustering_metrics(
    X,
    labels,
    model=None
):
    """
    Compute clustering metrics.
    """

    metrics = {}

    unique_labels = np.unique(labels)

    if len(unique_labels) > 1:

        try:

            metrics["Silhouette Score"] = (
                silhouette_score(
                    X,
                    labels
                )
            )

        except Exception:
            pass

        try:

            metrics["Davies-Bouldin Score"] = (
                davies_bouldin_score(
                    X,
                    labels
                )
            )

        except Exception:
            pass

        try:

            metrics["Calinski-Harabasz Score"] = (
                calinski_harabasz_score(
                    X,
                    labels
                )
            )

        except Exception:
            pass

    # Inertia (KMeans)

    try:

        if hasattr(model, "inertia_"):

            metrics["Inertia"] = (
                model.inertia_
            )

    except Exception:
        pass

    return metrics



# CLUSTERING METRICS DF


def clustering_metrics_dataframe(
    X,
    labels,
    model=None
):
    """
    Clustering metrics dataframe.
    """

    metrics = clustering_metrics(
        X,
        labels,
        model
    )

    return pd.DataFrame(
        list(metrics.items()),
        columns=["Metric", "Value"]
    )



# CONFUSION MATRIX DF


def confusion_matrix_dataframe(
    y_true,
    y_pred
):
    """
    Confusion matrix dataframe.
    """

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    return pd.DataFrame(cm)



# PREDICTION DF


def prediction_dataframe(
    y_true,
    y_pred
):
    """
    Actual vs Predicted dataframe.
    """

    return pd.DataFrame({
        "Actual": y_true,
        "Predicted": y_pred
    })



# FEATURE IMPORTANCE DF


def feature_importance_dataframe(
    feature_importance_dict
):
    """
    Convert feature importance
    dictionary to dataframe.
    """

    if feature_importance_dict is None:

        return pd.DataFrame()

    df = pd.DataFrame({

        "Feature":
            list(feature_importance_dict.keys()),

        "Importance":
            list(feature_importance_dict.values())
    })

    return df.sort_values(
        "Importance",
        ascending=False
    )



# EXPORT PACKAGE


def generate_export_bundle(
    metrics_df=None,
    report_df=None,
    cm_df=None,
    pred_df=None,
    importance_df=None
):
    """
    Package everything for Excel export.
    """

    return {
        "metrics": metrics_df,
        "classification_report": report_df,
        "confusion_matrix": cm_df,
        "predictions": pred_df,
        "feature_importance": importance_df
    }