
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, \
    recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.title("ML Classification Models Demo")

# Load trained models
# @st.cache_resource
def load_models():
    models = {}
    models["Logistic Regression"] = joblib.load("model/logistic_regression.pkl.gz")
    models["Decision Tree"] = joblib.load("model/decision_tree.pkl.gz")
    models["kNN"] = joblib.load("model/knn.pkl.gz")
    models["Naive Bayes"] = joblib.load("model/naive_bayes.pkl.gz")
    models["Random Forest"] = joblib.load("model/random_forest.pkl.gz")
    models["XGBoost"] = joblib.load("model/xgboost.pkl.gz")
    return models

models_dict = load_models()
model_name = st.selectbox("Select Model", list(models_dict.keys()))
selected_model = models_dict[model_name]

# Target column name (CHANGE TO YOURS)
target_col = "Churn"  # ← SAME AS TRAINING

# File upload
uploaded_file = st.file_uploader("Upload test CSV (must have same columns)", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.write("Preview:", test_df.head(3))

    if target_col not in test_df.columns:
        st.error(f"Missing target column '{target_col}'. Expected columns: {target_col}")
    else:
        X_test = test_df.drop(columns=[target_col])
        y_test_raw = test_df[target_col]

        # Encode target
        le_target = joblib.load("preprocessing/le_target.pkl.gz")
        y_test = le_target.transform(y_test_raw)

        # Run predictions and show metrics
        if st.button("Evaluate Model"):
            y_pred = selected_model.predict(X_test)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            mcc = matthews_corrcoef(y_test, y_pred)

            y_proba = selected_model.predict_proba(X_test)
            if len(le_target.classes_) == 2:
                auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_test, y_proba, multi_class='ovo', average='weighted')

            st.subheader("Evaluation Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{acc:.4f}")
                st.metric("Precision", f"{prec:.4f}")
            with col2:
                st.metric("Recall", f"{rec:.4f}")
                st.metric("F1 Score", f"{f1:.4f}")
            with col3:
                st.metric("AUC", f"{auc:.4f}")
                st.metric("MCC", f"{mcc:.4f}")

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            st.pyplot(fig)

            # Classification Report
            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred,
                                        target_names=le_target.classes_))
