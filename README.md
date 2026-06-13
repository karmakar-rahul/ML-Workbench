# ML Workbench

An interactive Machine Learning Dashboard built with Streamlit that enables users to perform Classification, Regression, and Clustering workflows through a no-code graphical interface. The application supports dataset exploration, preprocessing, model training, performance evaluation, visualization, and Excel report generation.

Streamlit App : [Try the interactive dashboard](https://ml-workbench.streamlit.app/)

---

## Project Overview

| Feature              | Description                                             |
| -------------------- | ------------------------------------------------------- |
| Dataset Exploration  | Load built-in datasets or upload custom datasets        |
| Data Preprocessing   | Missing value handling, encoding, scaling, and cleaning |
| Classification       | Train and evaluate multiple classification algorithms   |
| Regression           | Train and evaluate regression models                    |
| Clustering           | Perform unsupervised clustering analysis                |
| Visualization Studio | Interactive plots and performance visualizations        |
| Excel Export         | Export model results and evaluation reports             |
| Interactive UI       | Built using Streamlit for an intuitive workflow         |

### Project Architecture

| Module           | Responsibility                                      |
| ---------------- | --------------------------------------------------- |
| app.py           | Streamlit user interface and workflow orchestration |
| datasets.py      | Dataset loading and management                      |
| preprocessing.py | Data cleaning, encoding, scaling, and preprocessing |
| models.py        | Machine learning model registry and training        |
| metrics.py       | Evaluation metric computation                       |
| visualization.py | Plot generation and visual analytics                |
| export_excel.py  | Excel report creation and export                    |

### Project Structure

```text
ML-Workbench/
│
├── app.py
├── datasets.py
├── models.py
├── metrics.py
├── visualization.py
├── preprocessing.py
├── export_excel.py
│
├── README.md
├── requirements.txt
```
### Technology Stack

| Category                 | Technologies                |
| ------------------------ | --------------------------- |
| Language                 | Python                      |
| Frontend                 | Streamlit                   |
| Machine Learning         | Scikit-Learn, XGBoost       |
| Data Processing          | Pandas, NumPy               |
| Visualization            | Plotly, Matplotlib, Seaborn |
| Export Engine            | OpenPyXL                    |
| Dimensionality Reduction | PCA, UMAP                   |
| Explainability Ready     | SHAP                        |
---

## Supported Machine Learning Tasks

| Task Type                | Available |
| ------------------------ | --------- |
| Classification           | Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes, SVM, LDA, QDA, AdaBoost, Gradient Boosting, XGBoost, Bagging, Extra Trees        |
| Regression               |  Linear Regression, Ridge, Lasso, ElasticNet, Decision Tree Regressor, Random Forest Regressor, KNN Regressor, SVR, AdaBoost Regressor, Gradient Boosting Regressor, XGBoost Regressor, Bagging Regressor        |
| Clustering               |  KMeans, Agglomerative Clustering, DBSCAN, MeanShift, Birch, Gaussian Mixture Models        |

---

## Built-in Datasets

| Dataset            | Category       |
| ------------------ | -------------- |
| Iris               | Classification |
| Wine               | Classification |
| Breast Cancer      | Classification |
| Titanic            | Classification |
| Penguins           | Classification |
| Diabetes           | Regression     |
| California Housing | Regression     |
| MPG                | Regression     |
| Tips               | Regression     |
| Mall Customers     | Clustering     |


---

## Metrics

| Classification Metrics    |
| ------------------------- |
| Accuracy                  |
| Precision                 |
| Recall                    |
| F1 Score                  |
| Specificity               |
| Sensitivity               |
| True Positive Rate (TPR)  |
| False Positive Rate (FPR) |
| ROC-AUC Score             |
| Confusion Matrix          |

---

## Regression Metrics

| Metric                                |
| ------------------------------------- |
| Mean Absolute Error (MAE)             |
| Mean Squared Error (MSE)              |
| Root Mean Squared Error (RMSE)        |
| R² Score                              |
| Adjusted R²                           |
| Mean Absolute Percentage Error (MAPE) |

---

## Clustering Metrics

| Metric                  |
| ----------------------- |
| Silhouette Score        |
| Davies-Bouldin Index    |
| Calinski-Harabasz Index |
| Inertia                 |

---

## Visualization Features

### Classification

* Confusion Matrix
* ROC Curve
* Precision Recall Curve
* Feature Importance

### Regression

* Actual vs Predicted
* Residual Plot
* Error Distribution

### Clustering

* Cluster Visualization
* Elbow Method
* Silhouette Analysis

### General

* Correlation Heatmap
* PCA Projection

---

## Installation

### Clone Repository

```bash
git clone https://github.com/karmakar-rahul/ML-Workbench.git
cd ML-Workbench
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```
---

## Future Improvements

* Hyperparameter Optimization
* SHAP Explainability
* Model Comparison Dashboard
* AutoML Workflow
* Feature Selection Tools
* Time Series Analysis

---

## Author

Rahul Karmakar
* PGCP - BDA , C-DAC Chenai
* M.Sc. Physics (Astrophysics), Assam University
---

## License

This project is released under the MIT License.
