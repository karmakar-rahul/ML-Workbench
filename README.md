# ML Workbench

An interactive Machine Learning Visualization Dashboard built using Streamlit.

The application allows users to load datasets, train machine learning models, visualize results, compare model performance, and export detailed evaluation reports to Excel.

---

## Features

### Dataset Management

* Built-in datasets

  * Iris
  * Titanic
  * Wine
  * Breast Cancer
  * Diabetes
  * California Housing
  * Penguins
  * MPG
  * Tips
  * Mall Customers

* Custom CSV upload support

* Dataset summary

* Missing value analysis

* Data type inspection

---

## Machine Learning Tasks

### Classification

Supported Models:

* Logistic Regression
* Decision Tree
* Random Forest
* KNN
* Naive Bayes
* SVM
* LDA
* QDA
* AdaBoost
* Gradient Boosting
* XGBoost
* Bagging
* Extra Trees

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Specificity
* Sensitivity
* TPR
* FPR
* ROC-AUC
* Confusion Matrix

---

### Regression

Supported Models:

* Linear Regression
* Ridge
* Lasso
* ElasticNet
* Decision Tree Regressor
* Random Forest Regressor
* KNN Regressor
* SVR
* AdaBoost Regressor
* Gradient Boosting Regressor
* XGBoost Regressor
* Bagging Regressor

Evaluation Metrics:

* MAE
* MSE
* RMSE
* R²
* Adjusted R²
* MAPE

---

### Clustering

Supported Models:

* KMeans
* Agglomerative Clustering
* DBSCAN
* MeanShift
* Birch
* Gaussian Mixture Models

Evaluation Metrics:

* Silhouette Score
* Davies-Bouldin Index
* Calinski-Harabasz Score
* Inertia

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

## Excel Report Generation

Automatically exports:

* Dataset Summary
* Raw Data
* Train Data
* Test Data
* Predictions
* Metrics
* Classification Report
* Confusion Matrix
* Feature Importance
* Cross Validation Scores
* Model Parameters

---

## Installation

Clone repository:

```bash
git clone https://github.com/karmakar-rahul/ML-Workbench.git
cd ML-Workbench
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

---

## Project Structure

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

---

## Future Improvements

* Hyperparameter Optimization
* SHAP Explainability
* Model Comparison Dashboard
* AutoML Workflow
* Feature Selection Tools
* Time Series Analysis

---

## Technologies Used

* Python
* Streamlit
* Scikit-Learn
* XGBoost
* Pandas
* NumPy
* Plotly
* Matplotlib
* Seaborn
* OpenPyXL

---

## Author

Rahul Karmakar
PGCP - BDA , C-DAC Chenai
M.Sc. Physics (Astrophysics), Assam University
