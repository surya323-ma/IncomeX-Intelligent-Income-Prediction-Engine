# importing necessary libs
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)


# Sidebar
st.title("Income Classification - Full ML Model Pipeline")

st.sidebar.title("IncomeX")
st.sidebar.write("""
This project implements **6 machine learning models** on the **Adult Income Dataset**.
The goal is to **predict whether an individual earns >50K or <=50K** based on demographic and employment features.

**Instructions:**
1. Upload a test CSV file 
2. Select a machine learning model from the dropdown.
3. View predictions, evaluation metrics, confusion matrix, and classification report.
""")

# Load trained Models
@st.cache_data
def load_models():
    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        #"kNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
        #"Random Forest": joblib.load("model/random_forest.pkl"),
        "XGBoost": joblib.load("model/xgboost.pkl")
    }
    scaler = joblib.load("model/scaler.pkl")  # for models that require scaling
    feature_columns = joblib.load("model/feature_columns.pkl")  # saved during training
    return models, scaler, feature_columns

models, scaler, feature_columns = load_models()

# Upload Test CSV
st.header("Upload Test Dataset (CSV)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)

    # Check for target column
    if 'income' in test_data.columns:
        y_test = test_data['income'].map({">50K":1, "<=50K":0}).values
        X_test = test_data.drop('income', axis=1)
    else:
        y_test = None
        X_test = test_data.copy()

    # One-hot encode and align with trained features
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)

    st.success("Test dataset uploaded successfully!")
    st.write("Test Data Shape:", X_test.shape)
    
    # Model Selection
    st.header("Select Machine Learning Model")
    model_name = st.selectbox("Choose a model", list(models.keys()))
    model = models[model_name]

    # Scale features if required
    if model_name in ["Logistic Regression", "kNN", "Naive Bayes"]:
        X_input = scaler.transform(X_test)
    else:
        X_input = X_test.values

    # Predictions
    y_pred = model.predict(X_input)
    try:
        y_prob = model.predict_proba(X_input)[:, 1]
    except:
        y_prob = y_pred  # fallback if predict_proba not available

    # Show Metrics if Target Exists
    if y_test is not None:
        st.header("Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 3))
        col1.metric("Precision", round(precision_score(y_test, y_pred), 3))
        col2.metric("Recall", round(recall_score(y_test, y_pred), 3))
        col2.metric("F1 Score", round(f1_score(y_test, y_pred), 3))
        col3.metric("AUC", round(roc_auc_score(y_test, y_prob), 3))
        col3.metric("MCC", round(matthews_corrcoef(y_test, y_pred), 3))

        # Confusion Matrix
        st.header("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)

        # Classification Report
        st.header("Classification Report")
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))


    # Add Predictions to DataFrame & Download
    test_data['Predicted_Income'] = ["<=50K" if val==0 else ">50K" for val in y_pred]
    
else:
    st.info("Please upload a CSV file to proceed.")


# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Created by Raghunandan M S</p>",
    unsafe_allow_html=True
)
