"""
models.py

Machine Learning model management module
for ML Workbench Dashboard.

Supports:
- Classification
- Regression
- Clustering
- Dimensionality Reduction

"""

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor
)

from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor
)

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import (
    SVC,
    SVR
)

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN,
    MeanShift,
    Birch
)

from sklearn.mixture import GaussianMixture

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    from xgboost import (
        XGBClassifier,
        XGBRegressor
    )
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False



# CLASSIFICATION MODELS


CLASSIFICATION_MODELS = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(),

    "Naive Bayes":
        GaussianNB(),

    "SVM":
        SVC(
            probability=True,
            random_state=42
        ),

    "LDA":
        LinearDiscriminantAnalysis(),

    "QDA":
        QuadraticDiscriminantAnalysis(),

    "AdaBoost":
        AdaBoostClassifier(
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=42
        ),

    "Bagging":
        BaggingClassifier(
            random_state=42
        ),

    "Extra Trees":
        ExtraTreesClassifier(
            random_state=42
        )
}

if XGBOOST_AVAILABLE:
    CLASSIFICATION_MODELS["XGBoost"] = (
        XGBClassifier(
            eval_metric="logloss",
            random_state=42
        )
    )



# REGRESSION MODELS


REGRESSION_MODELS = {

    "Linear Regression":
        LinearRegression(),

    "Ridge":
        Ridge(),

    "Lasso":
        Lasso(),

    "ElasticNet":
        ElasticNet(),

    "Decision Tree Regressor":
        DecisionTreeRegressor(
            random_state=42
        ),

    "Random Forest Regressor":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),

    "KNN Regressor":
        KNeighborsRegressor(),

    "SVR":
        SVR(),

    "AdaBoost Regressor":
        AdaBoostRegressor(
            random_state=42
        ),

    "Gradient Boosting Regressor":
        GradientBoostingRegressor(
            random_state=42
        ),

    "Bagging Regressor":
        BaggingRegressor(
            random_state=42
        ),

    "Extra Trees Regressor":
        ExtraTreesRegressor(
            random_state=42
        )
}

if XGBOOST_AVAILABLE:
    REGRESSION_MODELS["XGBoost Regressor"] = (
        XGBRegressor(
            random_state=42
        )
    )



# CLUSTERING MODELS


CLUSTERING_MODELS = {

    "KMeans":
        KMeans(
            n_clusters=3,
            random_state=42
        ),

    "Agglomerative":
        AgglomerativeClustering(
            n_clusters=3
        ),

    "DBSCAN":
        DBSCAN(),

    "MeanShift":
        MeanShift(),

    "Birch":
        Birch(n_clusters=3),

    "Gaussian Mixture":
        GaussianMixture(
            n_components=3,
            random_state=42
        )
}



# DIMENSIONALITY REDUCTION


DIM_REDUCTION_MODELS = {

    "PCA":
        PCA(n_components=2),

    "t-SNE":
        TSNE(
            n_components=2,
            random_state=42
        )
}

if UMAP_AVAILABLE:

    DIM_REDUCTION_MODELS["UMAP"] = (
        UMAP(
            n_components=2,
            random_state=42
        )
    )



# GETTERS


def get_classification_models():
    return list(CLASSIFICATION_MODELS.keys())


def get_regression_models():
    return list(REGRESSION_MODELS.keys())


def get_clustering_models():
    return list(CLUSTERING_MODELS.keys())


def get_dimensionality_models():
    return list(DIM_REDUCTION_MODELS.keys())



# FACTORY FUNCTIONS


def get_classification_model(name):

    if name not in CLASSIFICATION_MODELS:
        raise ValueError(
            f"Classification model '{name}' not found."
        )

    return CLASSIFICATION_MODELS[name]


def get_regression_model(name):

    if name not in REGRESSION_MODELS:
        raise ValueError(
            f"Regression model '{name}' not found."
        )

    return REGRESSION_MODELS[name]


def get_clustering_model(name):

    if name not in CLUSTERING_MODELS:
        raise ValueError(
            f"Clustering model '{name}' not found."
        )

    return CLUSTERING_MODELS[name]


def get_dimensionality_model(name):

    if name not in DIM_REDUCTION_MODELS:
        raise ValueError(
            f"Dimensionality model '{name}' not found."
        )

    return DIM_REDUCTION_MODELS[name]



# TRAINING FUNCTIONS


def train_supervised_model(
    model,
    X_train,
    y_train
):
    """
    Fit classification/regression model.
    """

    model.fit(
        X_train,
        y_train
    )

    return model


def predict_supervised_model(
    model,
    X_test
):
    """
    Generate predictions.
    """

    return model.predict(X_test)



# CLASSIFICATION PROBABILITIES


def predict_probabilities(
    model,
    X_test
):
    """
    Predict probabilities if supported.
    """

    if hasattr(model, "predict_proba"):

        return model.predict_proba(X_test)

    return None



# CLUSTERING


def fit_clustering_model(
    model,
    X
):
    """
    Fit clustering model.
    """

    if isinstance(model, GaussianMixture):

        labels = model.fit_predict(X)

    else:

        labels = model.fit_predict(X)

    return labels



# DIMENSIONALITY REDUCTION


def perform_dimensionality_reduction(
    model,
    X
):
    """
    Apply PCA/tSNE/UMAP.
    """

    transformed = model.fit_transform(X)

    return transformed



# FEATURE IMPORTANCE


def get_feature_importance(
    model,
    feature_names
):
    """
    Return feature importance if supported.
    """

    if hasattr(model, "feature_importances_"):

        return {
            feature: importance
            for feature, importance
            in zip(
                feature_names,
                model.feature_importances_
            )
        }

    return None



# MODEL CATEGORIES


MODEL_CATEGORIES = {

    "Classification":
        get_classification_models(),

    "Regression":
        get_regression_models(),

    "Clustering":
        get_clustering_models(),

    "Dimensionality Reduction":
        get_dimensionality_models()
}