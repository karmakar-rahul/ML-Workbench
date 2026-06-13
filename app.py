"""
app.py

ML Workbench Dashboard

Sections:
- Dataset Loading
- Dataset Exploration
- Preprocessing
- Train/Test Split
- Model Training (Classification / Regression / Clustering)
- Evaluation Metrics & Visualizations
- Excel Export
"""

import tempfile

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

from datasets import (
    get_available_datasets,
    load_dataset,
    load_uploaded_csv,
    generate_dataset_overview,
    get_categorical_columns,
    fill_missing_values
)

from preprocessing import (
    detect_leakage_columns
)

from models import (
    get_classification_models,
    get_regression_models,
    get_clustering_models,
    get_classification_model,
    get_regression_model,
    get_clustering_model,
    train_supervised_model,
    predict_supervised_model,
    predict_probabilities,
    fit_clustering_model,
    get_feature_importance
)

from metrics import (
    classification_metrics,
    regression_metrics,
    clustering_metrics,

    classification_metrics_dataframe,
    regression_metrics_dataframe,
    clustering_metrics_dataframe,

    classification_report_dataframe,
    confusion_matrix_dataframe,

    prediction_dataframe,
    feature_importance_dataframe
)

# NOTE: the module is named "visualization.py" (singular).
from visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,

    plot_feature_importance,

    plot_actual_vs_predicted,
    plot_residuals,
    plot_error_distribution,

    plot_clusters,
    plot_pca_projection,
    plot_elbow_curve,
    plot_silhouette,
    plot_correlation_heatmap
)

from export_excel import (
    export_ml_report,
    export_regression_report,
    export_clustering_report
)



# THEORY / REFERENCE CONTENT (SIDEBAR)


THEORY = {
    "About this app": """
**ML Workbench** lets you load a dataset, choose a Classification,
Regression or Clustering task, train a model, and inspect the
resulting metrics, plots and an exportable Excel report.

Use this sidebar as a quick reference while you work - it does not
affect anything on the main page.
""",

    "Classification models": """
**Linear / probabilistic**
- *Logistic Regression* - linear decision boundary, fast, interpretable.
- *Naive Bayes* - assumes feature independence, works well on small/text data.
- *LDA / QDA* - assume (Gaussian) class distributions; QDA allows a
  separate covariance per class (more flexible, needs more data).

**Tree-based & ensembles**
- *Decision Tree* - simple, interpretable, prone to overfitting.
- *Random Forest / Extra Trees* - bagged trees, reduce overfitting,
  give feature importance.
- *Gradient Boosting / AdaBoost / XGBoost* - sequentially correct
  errors of previous trees, strong performance but can overfit if
  not tuned.
- *Bagging* - generic bootstrap-aggregation wrapper.

**Other**
- *KNN* - predicts using the majority class of nearby points;
  sensitive to feature scaling.
- *SVM* - finds a maximum-margin boundary; works well in
  high-dimensional spaces.
""",

    "Regression models": """
- *Linear Regression* - fits a straight-line relationship.
- *Ridge / Lasso / ElasticNet* - linear regression with
  regularization to reduce overfitting (Lasso can zero out features).
- *Decision Tree / Random Forest / Extra Trees Regressor* -
  non-linear, tree-based models.
- *Gradient Boosting / AdaBoost / Bagging / XGBoost Regressor* -
  ensemble methods, usually strong baselines.
- *KNN Regressor* - predicts the average of nearby points.
- *SVR* - support vector regression, margin-based.
""",

    "Clustering models": """
- *KMeans* - partitions data into *k* round-ish clusters by
  minimizing within-cluster variance.
- *Agglomerative* - builds a hierarchy of merges (bottom-up).
- *DBSCAN* - density-based; finds arbitrarily-shaped clusters and
  marks outliers as noise (label `-1`). Sensitive to `eps`.
- *MeanShift* - finds clusters by shifting points toward density
  peaks; no need to pre-specify *k*.
- *Birch* - incremental, memory-efficient clustering for large data.
- *Gaussian Mixture* - soft, probabilistic clustering assuming
  Gaussian-shaped clusters.
""",

    "Classification metrics": """
- **Accuracy** - fraction of correct predictions overall. Misleading
  on imbalanced data.
- **Precision** - of predicted positives, how many were correct.
- **Recall (Sensitivity / TPR)** - of actual positives, how many were
  found.
- **F1 Score** - harmonic mean of precision and recall.
- **Specificity / FPR** - true-negative and false-positive rates
  (binary problems).
- **ROC-AUC** - probability the model ranks a random positive above
  a random negative; 0.5 = random, 1.0 = perfect.
- **Confusion Matrix** - counts of TP/FP/TN/FN per class.

A score of ~100% is rarely realistic on real-world data and usually
signals **data leakage** (see "Data leakage" below) rather than a
great model.
""",

    "Regression metrics": """
- **MAE** - mean absolute error, same units as the target.
- **MSE / RMSE** - mean squared error / its square root; penalizes
  large errors more.
- **R²** - proportion of variance explained (1.0 = perfect,
  0 = no better than predicting the mean, can be negative).
- **Adjusted R²** - R² penalized for the number of features used.
- **MAPE** - mean absolute percentage error; unstable if the target
  contains values near zero.
""",

    "Clustering metrics": """
- **Silhouette Score** (-1 to 1) - how well-separated clusters are;
  higher is better.
- **Davies-Bouldin Score** - average similarity between clusters;
  lower is better.
- **Calinski-Harabasz Score** - ratio of between/within-cluster
  dispersion; higher is better.
- **Inertia** (KMeans only) - sum of squared distances to the
  nearest cluster center; used for the elbow method.
""",

    "Train/test split & cross-validation": """
The dataset is split into a **training set** (used to fit the model)
and a **test set** (used only to evaluate it). Metrics shown for
Classification/Regression come from this held-out test set.

**Cross-validation** (5-fold here) repeatedly re-splits the data and
re-trains the model to give a more stable estimate of performance and
a sense of variance across folds.

If both the test-set score *and* the cross-validation mean are very
high (and close to 1.0), that's a strong signal to check for data
leakage rather than celebrate.
""",

    "Feature scaling & encoding": """
- **Categorical encoding** - non-numeric columns are label-encoded
  (each category -> an integer) so models can use them.
- **Standard scaling** - rescales numeric features to mean 0,
  std 1. Important for distance-based models (KNN, SVM, clustering)
  and gradient-based linear models.

To avoid leaking information from the test set into the model, the
scaler is **fit on the training set only** and then applied to the
test set.
""",

    "Data leakage": """
**Data leakage** happens when information that wouldn't be available
at prediction time ends up in the features, causing unrealistically
high accuracy.

A classic example: the Titanic dataset's `alive` column is just a
re-encoding of `survived` (the usual target). Including it as a
feature lets a model "cheat" to ~100% accuracy.

This app automatically checks whether any feature almost perfectly
determines the target and excludes such columns from the *default*
feature selection (you'll see a warning listing them). You can still
add them back manually if you have a specific reason to.
"""
}



# HELPERS


def display_metrics(metrics_dict, n_cols=4):
    """
    Render a dict of metrics as st.metric widgets,
    handling numpy scalar types correctly.
    """

    cols = st.columns(min(n_cols, max(len(metrics_dict), 1)))

    for idx, (k, v) in enumerate(metrics_dict.items()):

        if isinstance(v, (np.integer,)):
            display_val = int(v)

        elif isinstance(v, (int, float, np.floating)):
            display_val = round(float(v), 4)

        else:
            display_val = str(v)

        cols[idx % len(cols)].metric(k, display_val)


def offer_excel_download(build_func, file_name, button_label, key):
    """
    Build an Excel report (via build_func, which writes to a path)
    and offer it for download.
    """

    if st.button(button_label, key=key):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".xlsx"
        ) as temp_file:

            build_func(temp_file.name)

            with open(temp_file.name, "rb") as f:
                st.download_button(
                    label=f"Download {file_name}",
                    data=f.read(),
                    file_name=file_name,
                    mime=(
                        "application/vnd.openxmlformats-"
                        "officedocument.spreadsheetml.sheet"
                    ),
                    key=f"{key}_download"
                )


def render_plot_grid(plot_specs):
    """
    Render a list of (kind, figure_or_none, empty_message) tuples
    in a responsive 2-column grid. `kind` is "pyplot" or "plotly".

    Figures that are None are shown as an `st.info` message instead
    of an empty/broken chart slot.
    """

    for i in range(0, len(plot_specs), 2):

        row = plot_specs[i:i + 2]

        cols = st.columns(len(row))

        for col, (kind, fig, empty_message) in zip(cols, row):

            with col:

                if fig is None:
                    st.info(empty_message)
                    continue

                if kind == "pyplot":
                    st.pyplot(fig)

                else:
                    st.plotly_chart(fig, use_container_width=True)



# PAGE CONFIG


st.set_page_config(
    page_title="ML Workbench",
    page_icon="📊",
    layout="wide"
)

st.title("Machine Learning Workbench")
st.markdown(
    "Interactive Machine Learning Visualization Dashboard"
)


# SIDEBAR - THEORY / REFERENCE ONLY


with st.sidebar:

    st.header("ML Reference")

    theory_topic = st.selectbox(
        "Choose a topic",
        list(THEORY.keys())
    )

    st.markdown(THEORY[theory_topic])



# DATASET SELECTION (MAIN PAGE)


st.header("1. Dataset")

source_col, picker_col = st.columns([1, 2])

with source_col:

    dataset_option = st.radio(
        "Dataset Source",
        [
            "Built-in Dataset",
            "Upload CSV"
        ]
    )

df = None

with picker_col:

    if dataset_option == "Built-in Dataset":

        dataset_name = st.selectbox(
            "Dataset",
            get_available_datasets()
        )

        try:
            df = load_dataset(dataset_name)

        except Exception as e:
            st.error(f"Could not load dataset: {e}")

    else:

        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"]
        )

        if uploaded_file:

            try:
                df = load_uploaded_csv(uploaded_file)

            except Exception as e:
                st.error(str(e))



# MAIN WORKFLOW


if df is None:

    st.info(
        "Select a built-in dataset or upload a CSV "
        "to get started."
    )

    st.stop()


st.success("Dataset Loaded Successfully")

overview = generate_dataset_overview(df)

st.subheader("Dataset Preview")
st.dataframe(df.head())


# SUMMARY METRICS


col1, col2, col3, col4 = st.columns(4)

col1.metric("Rows", overview["summary"]["Rows"])
col2.metric("Columns", overview["summary"]["Columns"])
col3.metric("Missing Values", overview["summary"]["Missing Values"])
col4.metric("Duplicate Rows", overview["summary"]["Duplicate Rows"])


# DETAILS


with st.expander("Dataset Summary"):
    st.dataframe(overview["statistics"])

with st.expander("Missing Value Report"):
    st.dataframe(overview["missing_report"])

with st.expander("Data Types"):
    st.dataframe(overview["dtype_report"])


# PREPROCESSING


st.header("2. Preprocessing")

auto_clean = st.checkbox(
    "Auto Fill Missing Values",
    value=True
)

if auto_clean:
    df = fill_missing_values(df)


# FEATURE / TARGET SELECTION


st.header("3. Feature Selection")

target_column = st.selectbox(
    "Target Column",
    df.columns
)

candidate_features = [
    col for col in df.columns if col != target_column
]

# --- Data leakage check -----------------------------------------
#
# Columns that almost perfectly determine the target (e.g. the
# Titanic "alive" column vs the "survived" target) are excluded from
# the *default* selection so models don't appear to score ~100% due
# to leaked information rather than genuine predictive power. They
# remain selectable manually.

leakage_info = detect_leakage_columns(
    df,
    target_column,
    candidate_features
)

default_features = [
    col for col in candidate_features if col not in leakage_info
]

if not default_features:
    default_features = candidate_features

if leakage_info:

    leakage_list = ", ".join(
        f"`{col}` ({purity:.0%} purity)"
        for col, purity in leakage_info.items()
    )

    st.warning(
        "The following column(s) almost perfectly determine the "
        "target and have been excluded from the default feature "
        f"selection to avoid unrealistic accuracy: {leakage_list}. "
        "You can re-add them manually below if intended."
    )

feature_columns = st.multiselect(
    "Feature Columns",
    candidate_features,
    default=default_features
)

if len(feature_columns) == 0:
    st.warning("Select at least one feature.")
    st.stop()

X = df[feature_columns].copy()
y = df[target_column].copy()


# ENCODE CATEGORICAL FEATURES


categorical_cols = get_categorical_columns(X)

feature_encoders = {}

for col in categorical_cols:

    encoder = LabelEncoder()

    X[col] = encoder.fit_transform(X[col].astype(str))

    feature_encoders[col] = encoder


# TARGET ENCODING


# "object", "category" and "bool" targets are treated as
# classification labels and encoded. `y` always ends up as a
# pandas Series with the *original* index so that, after
# train_test_split, y_train/y_test stay index-aligned with
# X_train/X_test (needed for clean Excel export sheets).
target_encoder = None

if str(y.dtype) in ("object", "category", "bool"):

    target_encoder = LabelEncoder()

    y = pd.Series(
        target_encoder.fit_transform(y.astype(str)),
        index=y.index,
        name=target_column
    )

else:
    y = y.rename(target_column)


# ML TASK & SCALING


st.header("4. Machine Learning Task")

ml_task = st.selectbox(
    "Select Task",
    [
        "Classification",
        "Regression",
        "Clustering"
    ]
)

use_scaling = st.checkbox(
    "Apply Standard Scaling",
    value=True
)


# TRAIN / TEST SPLIT (Classification / Regression)


X_train = X_test = y_train = y_test = None


def make_cv_estimator(model):
    """
    Wrap `model` in a scaling pipeline for cross-validation so that
    scaling statistics are computed per-fold (no leakage from the
    held-out fold into the scaler).
    """

    if use_scaling:
        return make_pipeline(StandardScaler(), model)

    return model


if ml_task != "Clustering":

    test_size = st.slider(
        "Test Size",
        min_value=0.10,
        max_value=0.50,
        value=0.20,
        step=0.05
    )

    # Stratify classification splits when every class has at least
    # 2 members, to keep class balance similar across train/test.
    stratify_target = None

    if ml_task == "Classification" and y.value_counts().min() >= 2:
        stratify_target = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify_target
    )

    if use_scaling:

        # Fit the scaler on the TRAINING data only, then apply the
        # same transform to the test data. Fitting on the full
        # dataset before splitting leaks test-set statistics into
        # training and gives optimistically biased results.
        scaler = StandardScaler()

        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

    st.success(f"Train Shape: {X_train.shape}")
    st.success(f"Test Shape: {X_test.shape}")

else:

    if use_scaling:

        scaler = StandardScaler()

        X = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )


# A scaled view of the full feature set, used only for the general
# PCA visualization at the bottom of the page (exploratory, not part
# of model evaluation).
if ml_task == "Clustering":
    X_for_viz = X

elif use_scaling:
    X_for_viz = pd.DataFrame(
        StandardScaler().fit_transform(X),
        columns=X.columns,
        index=X.index
    )

else:
    X_for_viz = X



# MODEL TRAINING SECTION


st.header("5. Model Selection")


# CLASSIFICATION


if ml_task == "Classification":

    model_name = st.selectbox(
        "Classification Model",
        get_classification_models()
    )

    if st.button("Train Classification Model"):

        model = get_classification_model(model_name)

        model = train_supervised_model(model, X_train, y_train)

        y_pred = predict_supervised_model(model, X_test)

        y_prob = predict_probabilities(model, X_test)

        metrics_dict = classification_metrics(y_test, y_pred, y_prob)
        metrics_df = classification_metrics_dataframe(y_test, y_pred, y_prob)

        report_df = classification_report_dataframe(y_test, y_pred)
        cm_df = confusion_matrix_dataframe(y_test, y_pred)

        importance_df = feature_importance_dataframe(
            get_feature_importance(model, X.columns)
        )

        prediction_df = prediction_dataframe(y_test, y_pred)

        # --------------------------------
        # CROSS VALIDATION
        # --------------------------------

        try:
            cv_model = make_cv_estimator(get_classification_model(model_name))
            cv_scores = cross_val_score(cv_model, X, y, cv=5)
            cv_mean = cv_scores.mean()

        except Exception:
            cv_scores = None
            cv_mean = None

        # --------------------------------
        # METRICS DISPLAY
        # --------------------------------

        st.subheader("Classification Metrics")

        display_metrics(metrics_dict)

        if cv_mean is not None:
            st.metric("CV Mean Score (5-fold)", round(cv_mean, 4))

        if metrics_dict.get("Accuracy", 0) >= 0.999 or (
            cv_mean is not None and cv_mean >= 0.999
        ):
            st.warning(
                "Accuracy is at (or extremely close to) 100%. "
                "Real-world datasets almost never support this - "
                "double check the Feature Selection step for any "
                "remaining leakage columns."
            )

        with st.expander("Classification Report"):
            st.dataframe(report_df)

        # --------------------------------
        # VISUALIZATIONS
        # --------------------------------

        st.subheader("Visualizations")

        render_plot_grid([
            (
                "pyplot",
                plot_confusion_matrix(y_test, y_pred),
                "Confusion matrix unavailable."
            ),
            (
                "plotly",
                plot_roc_curve(y_test, y_prob),
                "ROC curve is only available for binary "
                "classification with probability outputs."
            )
        ])

        importance_fig = None

        if len(importance_df):
            importance_fig = plot_feature_importance(
                dict(zip(importance_df["Feature"], importance_df["Importance"]))
            )

        render_plot_grid([
            (
                "plotly",
                plot_precision_recall_curve(y_test, y_prob),
                "Precision-Recall curve is only available for "
                "binary classification with probability outputs."
            ),
            (
                "plotly",
                importance_fig,
                "This model does not expose feature importances."
            )
        ])

        # --------------------------------
        # CACHE RESULTS FOR EXPORT
        # --------------------------------

        st.session_state["clf_results"] = {
            "raw_data": df,
            "train_data": pd.concat([X_train, y_train], axis=1),
            "test_data": pd.concat([X_test, y_test], axis=1),
            "predictions": prediction_df,
            "metrics_df": metrics_df,
            "report_df": report_df,
            "cm_df": cm_df,
            "importance_df": importance_df,
            "cv_scores": cv_scores,
            "summary": overview["summary"],
            "model": model
        }

    # --------------------------------
    # EXPORT
    # --------------------------------

    if "clf_results" in st.session_state:

        results = st.session_state["clf_results"]

        offer_excel_download(
            build_func=lambda path: export_ml_report(
                output_path=path,
                raw_data=results["raw_data"],
                train_data=results["train_data"],
                test_data=results["test_data"],
                predictions=results["predictions"],
                metrics_df=results["metrics_df"],
                classification_report_df=results["report_df"],
                confusion_matrix_df=results["cm_df"],
                feature_importance_df=results["importance_df"],
                cross_validation_scores=results["cv_scores"],
                dataset_summary=results["summary"],
                model=results["model"]
            ),
            file_name="ML_Classification_Report.xlsx",
            button_label="Export Excel Report",
            key="export_classification"
        )


# REGRESSION


elif ml_task == "Regression":

    model_name = st.selectbox(
        "Regression Model",
        get_regression_models()
    )

    if st.button("Train Regression Model"):

        model = get_regression_model(model_name)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics_dict = regression_metrics(y_test, y_pred, X.shape[1])
        metrics_df = regression_metrics_dataframe(y_test, y_pred, X.shape[1])

        prediction_df = prediction_dataframe(y_test, y_pred)

        importance_df = feature_importance_dataframe(
            get_feature_importance(model, X.columns)
        )

        # --------------------------------
        # CROSS VALIDATION
        # --------------------------------

        try:
            cv_model = make_cv_estimator(get_regression_model(model_name))
            cv_scores = cross_val_score(cv_model, X, y, cv=5)

        except Exception:
            cv_scores = None

        # --------------------------------
        # METRICS DISPLAY
        # --------------------------------

        st.subheader("Regression Metrics")

        display_metrics(metrics_dict)

        if cv_scores is not None:
            st.metric("CV Mean R2 (5-fold)", round(cv_scores.mean(), 4))

        # --------------------------------
        # VISUALIZATIONS
        # --------------------------------

        st.subheader("Visualizations")

        render_plot_grid([
            (
                "plotly",
                plot_actual_vs_predicted(y_test, y_pred),
                "Actual vs Predicted plot unavailable."
            ),
            (
                "plotly",
                plot_residuals(y_test, y_pred),
                "Residual plot unavailable."
            )
        ])

        importance_fig = None

        if len(importance_df):
            importance_fig = plot_feature_importance(
                dict(zip(importance_df["Feature"], importance_df["Importance"]))
            )

        render_plot_grid([
            (
                "plotly",
                plot_error_distribution(y_test, y_pred),
                "Error distribution plot unavailable."
            ),
            (
                "plotly",
                importance_fig,
                "This model does not expose feature importances."
            )
        ])

        # --------------------------------
        # CACHE RESULTS FOR EXPORT
        # --------------------------------

        st.session_state["reg_results"] = {
            "raw_data": df,
            "train_data": pd.concat([X_train, y_train], axis=1),
            "test_data": pd.concat([X_test, y_test], axis=1),
            "predictions": prediction_df,
            "metrics_df": metrics_df,
            "importance_df": importance_df,
            "cv_scores": cv_scores,
            "summary": overview["summary"],
            "model": model
        }

    # --------------------------------
    # EXPORT
    # --------------------------------

    if "reg_results" in st.session_state:

        results = st.session_state["reg_results"]

        offer_excel_download(
            build_func=lambda path: export_regression_report(
                output_path=path,
                raw_data=results["raw_data"],
                train_data=results["train_data"],
                test_data=results["test_data"],
                predictions=results["predictions"],
                metrics_df=results["metrics_df"],
                feature_importance_df=results["importance_df"],
                dataset_summary=results["summary"],
                model=results["model"],
                cv_scores=results["cv_scores"]
            ),
            file_name="ML_Regression_Report.xlsx",
            button_label="Export Excel Report",
            key="export_regression"
        )


# CLUSTERING


elif ml_task == "Clustering":

    model_name = st.selectbox(
        "Clustering Model",
        get_clustering_models()
    )

    if st.button("Run Clustering"):

        model = get_clustering_model(model_name)

        labels = fit_clustering_model(model, X)

        metrics_dict = clustering_metrics(X, labels, model)
        metrics_df = clustering_metrics_dataframe(X, labels, model)

        st.subheader("Clustering Metrics")

        display_metrics(metrics_dict)

        # --------------------------------
        # VISUALIZATIONS
        # --------------------------------

        st.subheader("Visualizations")

        render_plot_grid([
            (
                "plotly",
                plot_clusters(X, labels),
                "Cluster scatter plot needs at least 2 numeric "
                "features."
            ),
            (
                "pyplot",
                plot_silhouette(X, labels),
                "Silhouette plot is unavailable: the chosen model "
                "produced fewer than 2 clusters."
            )
        ])

        render_plot_grid([
            (
                "plotly",
                plot_elbow_curve(X),
                "Elbow curve unavailable."
            )
        ])

        # --------------------------------
        # CACHE RESULTS FOR EXPORT
        # --------------------------------

        st.session_state["clu_results"] = {
            "raw_data": df,
            "metrics_df": metrics_df,
            "summary": overview["summary"],
            "model": model
        }

    # --------------------------------
    # EXPORT
    # --------------------------------

    if "clu_results" in st.session_state:

        results = st.session_state["clu_results"]

        offer_excel_download(
            build_func=lambda path: export_clustering_report(
                output_path=path,
                raw_data=results["raw_data"],
                metrics_df=results["metrics_df"],
                dataset_summary=results["summary"],
                model=results["model"]
            ),
            file_name="ML_Clustering_Report.xlsx",
            button_label="Export Excel Report",
            key="export_clustering"
        )



# GENERAL VISUALIZATION SECTION


st.header("Exploratory Visualizations")

try:
    corr_fig = plot_correlation_heatmap(df)

except Exception:
    corr_fig = None

pca_fig = plot_pca_projection(X_for_viz)

render_plot_grid([
    (
        "pyplot",
        corr_fig,
        "Correlation heatmap needs at least one numeric column."
    ),
    (
        "plotly",
        pca_fig,
        "PCA projection needs at least 2 selected features."
    )
])



# FOOTER


st.markdown("---")

st.caption(
    "ML Workbench Dashboard | Classification • Regression • Clustering"
)
