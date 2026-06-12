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

from datasets import (
    get_available_datasets,
    load_dataset,
    load_uploaded_csv,
    generate_dataset_overview,
    get_categorical_columns,
    fill_missing_values
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

# DATASET SELECTION
st.header("Dataset Selection")

dataset_option = st.radio(
    "Choose Dataset Source",
    [
        "Built-in Dataset",
        "Upload CSV"
    ]
)
# LOAD DATA
df = None

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
st.header("Preprocessing")

auto_clean = st.checkbox(
    "Auto Fill Missing Values",
    value=True
)

if auto_clean:
    df = fill_missing_values(df)


# FEATURE / TARGET SELECTION
st.header("Feature Selection")

target_column = st.selectbox(
    "Target Column",
    df.columns
)

feature_columns = st.multiselect(
    "Feature Columns",
    [col for col in df.columns if col != target_column],
    default=[col for col in df.columns if col != target_column]
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
# Treat "object", "category" and "bool" targets as classification
# labels and encode them. The previous check (`y.dtype == "object"`)
# missed pandas "category" dtype columns (common in the
# Titanic/Penguins datasets, e.g. "class", "deck"), which could be
# passed unencoded into sklearn estimators and cause fit() to fail.
target_encoder = None
if str(y.dtype) in ("object", "category", "bool"):
    target_encoder = LabelEncoder()

    # Keep the result as a pandas Series with the *original* index.
    # This matters later: train_test_split keeps X/y indices aligned,
    # which is what allows train/test sheets in the Excel export to
    # be built with a simple pd.concat (no NaNs from index mismatch).
    y = pd.Series(
        target_encoder.fit_transform(y.astype(str)),
        index=y.index,
        name=target_column
    )

else:
    y = y.rename(target_column)


# SCALING
use_scaling = st.checkbox(
    "Apply Standard Scaling",
    value=True
)
if use_scaling:
    scaler = StandardScaler()
    X = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

# PROBLEM TYPE
st.header("Machine Learning Task")
ml_task = st.selectbox(
    "Select Task",
    [
        "Classification",
        "Regression",
        "Clustering"
    ]
)

# TRAIN / TEST SPLIT
X_train = X_test = y_train = y_test = None
if ml_task != "Clustering":
    test_size = st.slider(
        "Test Size",
        min_value=0.10,
        max_value=0.50,
        value=0.20,
        step=0.05
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )
    st.success(f"Train Shape: {X_train.shape}")
    st.success(f"Test Shape: {X_test.shape}")


# MODEL TRAINING SECTION
st.header("Model Selection")


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

  
        # CROSS VALIDATION
        try:
            cv_scores = cross_val_score(model, X, y, cv=5)
            cv_mean = cv_scores.mean()

        except Exception:
            cv_scores = None
            cv_mean = None


        # METRICS DISPLAY
        st.subheader("Classification Metrics")

        display_metrics(metrics_dict)

        if cv_mean is not None:
            st.metric("CV Mean Score (5-fold)", round(cv_mean, 4))

        with st.expander("Classification Report"):
            st.dataframe(report_df)

        # VISUALIZATIONS
        st.subheader("Confusion Matrix")
        st.pyplot(plot_confusion_matrix(y_test, y_pred))

        roc_fig = plot_roc_curve(y_test, y_prob)

        if roc_fig is not None:
            st.plotly_chart(roc_fig, use_container_width=True)

        pr_fig = plot_precision_recall_curve(y_test, y_prob)

        if pr_fig is not None:
            st.plotly_chart(pr_fig, use_container_width=True)

        if len(importance_df):

            st.subheader("Feature Importance")

            st.plotly_chart(
                plot_feature_importance(
                    dict(zip(importance_df["Feature"], importance_df["Importance"]))
                ),
                use_container_width=True
            )


        # CACHE RESULTS FOR EXPORT

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


    # EXPORT
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


        # CROSS VALIDATION
        try:
            cv_scores = cross_val_score(model, X, y, cv=5)

        except Exception:
            cv_scores = None

   
        # METRICS DISPLAY
      

        st.subheader("Regression Metrics")

        display_metrics(metrics_dict)

        if cv_scores is not None:
            st.metric("CV Mean R2 (5-fold)", round(cv_scores.mean(), 4))

      
        # VISUALIZATIONS
        st.plotly_chart(
            plot_actual_vs_predicted(y_test, y_pred),
            use_container_width=True
        )

        st.plotly_chart(
            plot_residuals(y_test, y_pred),
            use_container_width=True
        )

        st.plotly_chart(
            plot_error_distribution(y_test, y_pred),
            use_container_width=True
        )

        if len(importance_df):

            st.subheader("Feature Importance")

            st.plotly_chart(
                plot_feature_importance(
                    dict(zip(importance_df["Feature"], importance_df["Importance"]))
                ),
                use_container_width=True
            )

        # CACHE RESULTS FOR EXPORT
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


    # EXPORT
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

    
        # VISUALIZATIONS
        cluster_fig = plot_clusters(X, labels)

        if cluster_fig is not None:
            st.plotly_chart(cluster_fig, use_container_width=True)
        else:
            st.info(
                "Cluster scatter plot needs at least 2 numeric "
                "features."
            )

        silhouette_fig = plot_silhouette(X, labels)

        if silhouette_fig is not None:
            st.pyplot(silhouette_fig)
        else:
            st.info(
                "Silhouette plot is unavailable: the chosen "
                "model produced fewer than 2 clusters."
            )

        elbow_fig = plot_elbow_curve(X)
        st.plotly_chart(elbow_fig, use_container_width=True)

        # CACHE RESULTS FOR EXPORT
     

        st.session_state["clu_results"] = {
            "raw_data": df,
            "metrics_df": metrics_df,
            "summary": overview["summary"],
            "model": model
        }

    # EXPORT
   

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
    st.pyplot(plot_correlation_heatmap(df))

except Exception:
    pass

pca_fig = plot_pca_projection(X)

if pca_fig is not None:
    st.plotly_chart(pca_fig, use_container_width=True)


# FOOTER

st.markdown("---")

st.caption(
    "ML Workbench Dashboard | Classification • Regression • Clustering"
)