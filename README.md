# IncomeX – Intelligent Income Prediction Engine
A complete ML Pipe Program on Income Classification Problem including Streamlit

a. Problem Statement
The goal of this project is to predict whether an individual earns more than 50K (>50K) or less than or equal to 50K (<=50K) per year based on demographic and employment attributes.
This is a binary classification problem where various machine learning models are compared in terms of performance metrics.

The project also includes a Streamlit app for uploading test datasets, evaluating models, and visualizing results.

b. Dataset Description
The Adult Income Dataset is sourced from the UCI Machine Learning Repository.
It contains 48,842 instances and 14 features (after preprocessing and one-hot encoding).
The features include both numerical and categorical attributes related to an individual's age, workclass, education, occupation, marital status, and hours worked per week.
The target variable is income, which indicates whether the individual earns >50K or <=50K.

data = fetch_ucirepo(id=2) df = data.data.original df.head()

Key features:

Feature Name	Description
age	Age of the individual
workclass	Employment type
education-num	Number of years of education
marital-status	Marital status of the individual
occupation	Type of occupation
relationship	Family relationship
race	Race of the individual
sex	Gender
capital-gain	Income from capital gains
capital-loss	Loss from capital
hours-per-week	Work hours per week
native-country	Country of origin
c. Models used
Handles datasets with 14 features (numerical + categorical with one-hot encoding).

Supports 6 machine learning algorithms:

Logistic Regression
Decision Tree Classifier
k-Nearest Neighbors (kNN)
Naive Bayes (Gaussian)
Random Forest (Ensemble)
XGBoost (Ensemble)
Provides evaluation metrics:

Accuracy
Precision
Recall
F1 Score
AUC Score
Matthews Correlation Coefficient (MCC)
Displays:

Confusion Matrix
Classification Report in table format
Predictions for download
Streamlit UI:

File uploader (CSV)
Model selection dropdown
Metrics and visualization display
Download predictions
Comparison Table of Evaluation Metrics
ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.85	0.91	0.78	0.62	0.69	0.62
Decision Tree	0.82	0.85	0.71	0.64	0.67	0.58
kNN	0.84	0.88	0.75	0.61	0.67	0.60
Naive Bayes	0.79	0.86	0.71	0.53	0.61	0.52
Random Forest	0.86	0.93	0.80	0.64	0.71	0.65
XGBoost	0.87	0.94	0.82	0.65	0.73	0.67
Observations on Model Performance
ML Model Name	Observation about model performance
Logistic Regression	Performs reasonably well with balanced accuracy; handles linear relationships but may underperform on non-linear interactions.
Decision Tree	Good at capturing non-linear patterns but prone to overfitting; slightly lower generalization performance.
kNN	Sensitive to feature scaling and neighbors; performs moderately well but slower on large datasets.
Naive Bayes	Fast and simple; assumes feature independence which may not hold, resulting in lower recall.
Random Forest	Ensemble approach improves generalization; robust and highest recall among traditional models.
XGBoost	Best overall performance; handles non-linearity and interactions well, achieving highest AUC and MCC.
Note:
Please note that the dataset is trained on kNN and RF models, however, the models are >25 mb, it will not be displayed under the drop down menu.

Deployment:
The project is deployed using Streamlit Community Cloud.

`https://icmlbits.streamlit.app'

Deployment Steps:

Push the project folder to a GitHub repository.
Go to Streamlit Cloud and connect your GitHub account.
Select the repository and branch, then deploy.
Users can now upload CSV files, select a model, and get predictions with evaluation metrics in real time.
Installation:
Clone the repository:

git clone
cd project-folder
Create virtual enviroment (optional):
python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows
Usage:
streamlit run app.py

Upload a CSV file with features.
Target column income is optional.
Select a machine learning model.
View predictions and evaluation metrics (if income is provided).
Download the predictions as a CSV.
Folder Structure:
project-folder/
│
├── app.py                     # Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── model/                     # Saved ML models and artifacts
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
└── data/ (optional)
    └── sample_test_data.csv   # Sample CSV for testing (without target)
Author:
Raghunandan M S Created as part of a machine learning classification project @ BiTS Pilani.

References
UCI Adult Income Dataset
Scikit-learn Documentation
XGBoost Documentation
Streamlit Documentation
