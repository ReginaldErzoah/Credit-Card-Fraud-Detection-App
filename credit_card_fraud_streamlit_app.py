# -----------------------------
# Patch NumPy aliases for SHAP
# -----------------------------
import numpy as np

# Add all deprecated aliases SHAP might use
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "object"):
    np.object = object
if not hasattr(np, "str"):
    np.str = str

# Now import all other libraries
import streamlit as st
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# -----------------------------
# Load deployment objects
# -----------------------------
pkl_path = Path("fraud_detection_deployment_objects.pkl")

if not pkl_path.exists():
    st.error(f"Deployment file not found! Expected at: {pkl_path.resolve()}")
    st.stop()

deployment_objects = joblib.load(pkl_path)

lr = deployment_objects.get("logreg")
rf = deployment_objects.get("rf")
xgb_model = deployment_objects.get("xgb")
scaler = deployment_objects.get("scaler")
feature_names = deployment_objects.get("feature_names")

# -----------------------------
# Create SHAP explainer
# -----------------------------
shap_xgb = None

if xgb_model is not None:
    try:
        import shap
        shap_xgb = shap.TreeExplainer(xgb_model)
    except Exception as e:
        st.warning(f"SHAP explainer could not be created: {e}")

# -----------------------------
# App title
# -----------------------------
st.title("Credit Card Fraud Detection Dashboard")

st.write(
"""
This application predicts whether a credit card transaction is fraudulent
using multiple machine learning models and provides explainability using SHAP.
"""
)

# -----------------------------
# Model selection
# -----------------------------
model_choice = st.selectbox(
    "Select model:",
    ["Logistic Regression", "Random Forest", "XGBoost"]
)

model_map = {
    "Logistic Regression": lr,
    "Random Forest": rf,
    "XGBoost": xgb_model
}

model = model_map[model_choice]

# -----------------------------
# Threshold input
# -----------------------------
threshold_input = st.text_input(
    "Set prediction threshold (0.0 - 1.0)",
    value="0.5"
)

try:
    threshold = float(threshold_input)

    if not 0 <= threshold <= 1:
        st.error("Threshold must be between 0 and 1")
        st.stop()

except ValueError:
    st.error("Threshold must be numeric")
    st.stop()

st.write(f"Current threshold: {threshold:.6f}")

# -----------------------------
# Input method
# -----------------------------
st.subheader("Input Transaction Data")

input_option = st.radio(
    "Choose input method:",
    ["Manual Entry", "Upload CSV"]
)

# Default features if missing
feature_names = feature_names or ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']

# -----------------------------
# Manual Input
# -----------------------------
if input_option == "Manual Entry":

    input_data = {
        feature: st.number_input(feature, value=0.0)
        for feature in feature_names
    }

    input_df = pd.DataFrame([input_data])

# -----------------------------
# CSV Upload
# -----------------------------
else:

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is None:
        st.stop()

    input_df = pd.read_csv(uploaded_file)

    input_df.columns = input_df.columns.str.strip()

    st.write("Preview of uploaded data")
    st.dataframe(input_df.head())

    missing_cols = [c for c in feature_names if c not in input_df.columns]

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    extra_cols = [c for c in input_df.columns if c not in feature_names]

    if extra_cols:
        st.warning(f"Extra columns will be ignored: {extra_cols}")

    input_df = input_df[feature_names].astype(float)

# -----------------------------
# Feature Engineering
# -----------------------------
input_df["Hour"] = (input_df["Time"] // 3600) % 24

feature_names_with_hour = feature_names + ["Hour"]

input_df = input_df[feature_names_with_hour]

# -----------------------------
# Prepare Model Input
# -----------------------------
if model_choice == "Logistic Regression":

    model_input = input_df[feature_names]

    scaled_input = scaler.transform(model_input)

else:

    model_input = input_df[feature_names_with_hour]

    scaled_input = model_input.values

# -----------------------------
# Make Predictions
# -----------------------------
pred_probs = model.predict_proba(scaled_input)[:,1]

pred_classes = (pred_probs >= threshold).astype(int)

# -----------------------------
# Display Results
# -----------------------------
results = input_df.copy()

results["Fraud_Probability"] = pred_probs
results["Predicted_Class"] = pred_classes

st.subheader("Prediction Results")

st.dataframe(results)

# -----------------------------
# Fraud Probability Distribution
# -----------------------------
st.subheader("Fraud Probability Distribution")

fig, ax = plt.subplots()

ax.hist(pred_probs, bins=30)

ax.set_xlabel("Fraud Probability")
ax.set_ylabel("Number of Transactions")
ax.set_title("Distribution of Fraud Predictions")

st.pyplot(fig)

# -----------------------------
# SHAP Global Feature Importance
# -----------------------------
if model_choice == "XGBoost" and shap_xgb is not None:

    st.subheader("Global Feature Importance (SHAP)")

    sample_input = input_df[feature_names_with_hour].head(min(100, len(input_df)))

    shap_values = shap_xgb.shap_values(sample_input)

    import shap

    fig, ax = plt.subplots(figsize=(10,5))

    shap.summary_plot(
        shap_values,
        sample_input,
        show=False
    )

    st.pyplot(fig)

# -----------------------------
# SHAP Individual Explanation
# -----------------------------
if model_choice == "XGBoost" and shap_xgb is not None:

    st.subheader("Explain Individual Prediction")

    transaction_index = st.number_input(
        "Select transaction index",
        min_value=0,
        max_value=len(input_df)-1,
        value=0
    )

    transaction = input_df.iloc[[transaction_index]]

    shap_values_single = shap_xgb.shap_values(transaction)

    import shap

    fig, ax = plt.subplots(figsize=(10,4))

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_single[0],
            base_values=shap_xgb.expected_value,
            data=transaction.iloc[0],
            feature_names=transaction.columns
        ),
        show=False
    )

    st.pyplot(fig)

# -----------------------------
# Download Predictions
# -----------------------------
csv = results.to_csv(index=False).encode()

st.download_button(
    label="Download Predictions as CSV",
    data=csv,
    file_name="fraud_predictions.csv",
    mime="text/csv"
)






