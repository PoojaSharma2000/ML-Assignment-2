import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="wide"
)


# ---------------------------------------------------------
# Load Models
# ---------------------------------------------------------

@st.cache_resource
def load_models():

    models = {
        "Logistic Regression": joblib.load(
            "model/logistic_regression.pkl"
        ),

        "Decision Tree": joblib.load(
            "model/decision_tree.pkl"
        ),

        "KNN": joblib.load(
            "model/knn.pkl"
        ),

        "Naive Bayes": joblib.load(
            "model/naive_bayes.pkl"
        ),

        "Random Forest": joblib.load(
            "model/random_forest.pkl"
        )
    }

    scaler = joblib.load("model/scaler.pkl")

    return models, scaler


models, scaler = load_models()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🔬 Breast Cancer Classification")

st.write(
    "Interactive machine learning application for comparing "
    "multiple classification models on the UCI Breast Cancer "
    "Wisconsin (Diagnostic) dataset."
)

st.info(
    "Upload the test dataset generated during model evaluation "
    "to view predictions and performance metrics."
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Select Classification Model",
    list(models.keys())
)


# ---------------------------------------------------------
# Dataset Upload
# ---------------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)


# ---------------------------------------------------------
# Main Application
# ---------------------------------------------------------

if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file)

    st.success("Test dataset uploaded successfully.")

    st.subheader("Dataset Preview")

    st.dataframe(
        test_data.head(),
        use_container_width=True
    )

    # Check target column
    if "diagnosis" not in test_data.columns:

        st.error(
            "The uploaded CSV must contain a 'diagnosis' column."
        )

        st.stop()

    # Separate features and target
    X_test = test_data.drop(columns=["diagnosis"])
    y_test = test_data["diagnosis"]

    # Convert target if required
    if y_test.dtype == "object":

        y_test = y_test.map({
            "B": 0,
            "M": 1
        })

    # Select model
    model = models[selected_model]

    # Apply scaling for models that require it
    if selected_model in [
        "Logistic Regression",
        "KNN"
    ]:

        X_processed = scaler.transform(X_test)

    else:

        X_processed = X_test

    # Predictions
    y_pred = model.predict(X_processed)

    y_prob = model.predict_proba(
        X_processed
    )[:, 1]

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    st.header("2. Evaluation Metrics")

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

    col2.metric(
        "AUC",
        f"{auc:.4f}"
    )

    col3.metric(
        "Precision",
        f"{precision:.4f}"
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Recall",
        f"{recall:.4f}"
    )

    col5.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

    col6.metric(
        "MCC",
        f"{mcc:.4f}"
    )


    # -----------------------------------------------------
    # Confusion Matrix
    # -----------------------------------------------------

    st.header("3. Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Benign", "Actual Malignant"],
        columns=["Predicted Benign", "Predicted Malignant"]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )


    # -----------------------------------------------------
    # Classification Report
    # -----------------------------------------------------

    st.header("4. Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "Benign",
            "Malignant"
        ],
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )


    # -----------------------------------------------------
    # Prediction Summary
    # -----------------------------------------------------

    st.header("5. Prediction Summary")

    prediction_counts = pd.Series(
        y_pred
    ).value_counts()

    summary_df = pd.DataFrame({
        "Class": [
            "Benign (0)",
            "Malignant (1)"
        ],
        "Predictions": [
            prediction_counts.get(0, 0),
            prediction_counts.get(1, 0)
        ]
    })

    st.dataframe(
        summary_df,
        use_container_width=True
    )

else:

    st.warning(
        "Please upload test_data.csv to view model results."
    )
