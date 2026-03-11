# -----------------------------
# Import libraries
# -----------------------------
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
import shap  # Modern SHAP API

# -----------------------------
# Load deployment objects
# -----------------------------

# Load XGBoost JSON model
xgb_model = XGBClassifier()
xgb_model.load_model("xgb_model.json")

pkl_path = Path("fraud_detection_deployment_objects.pkl")
if not pkl_path.exists():
    st.error(f"Deployment file not found! Expected at: {pkl_path.resolve()}")
    st.stop()

deployment_objects = joblib.load(pkl_path)
lr = deployment_objects.get("logreg")
rf = deployment_objects.get("rf")
xgb_model = deployment_objects.get("xgb")
scaler = deployment_objects.get("scaler")
feature_names = deployment_objects.get("feature_names") or ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']

# -----------------------------
# App title
# -----------------------------
st.title("Credit Card Fraud Detection Dashboard")
st.write("""
This application predicts whether a credit card transaction is fraudulent
using multiple machine learning models and provides explainability using SHAP.
""")

# -----------------------------
# Model selection
# -----------------------------
model_choice = st.selectbox("Select model:", ["Logistic Regression", "Random Forest", "XGBoost"])
model_map = {"Logistic Regression": lr, "Random Forest": rf, "XGBoost": xgb_model}
model = model_map[model_choice]

# -----------------------------
# Threshold input
# -----------------------------
threshold_input = st.text_input("Set prediction threshold (0.0 - 1.0)", value="0.5")
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
input_option = st.radio("Choose input method:", ["Manual Entry", "Upload CSV"])

if input_option == "Manual Entry":
    input_data = {feature: st.number_input(feature, value=0.0) for feature in feature_names}
    input_df = pd.DataFrame([input_data])
else:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
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
# Feature engineering
# -----------------------------
input_df["Hour"] = (input_df["Time"] // 3600) % 24
feature_names_with_hour = feature_names + ["Hour"]
input_df = input_df[feature_names_with_hour]

# -----------------------------
# Prepare model input
# -----------------------------
if model_choice == "Logistic Regression":
    model_input = input_df[feature_names]
    scaled_input = scaler.transform(model_input)
else:
    model_input = input_df[feature_names_with_hour]
    scaled_input = model_input.values

# -----------------------------
# Make predictions
# -----------------------------
pred_probs = model.predict_proba(scaled_input)[:, 1]
pred_classes = (pred_probs >= threshold).astype(int)

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
# SHAP Explainability (Modern API)
# -----------------------------
if model_choice == "XGBoost" and xgb_model is not None:
    st.subheader("Global Feature Importance (SHAP)")
    shap_explainer = shap.Explainer(xgb_model, model_input)
    shap_values = shap_explainer(model_input)

    fig, ax = plt.subplots(figsize=(10,5))
    shap.summary_plot(shap_values.values, model_input, show=False)
    st.pyplot(fig)

    st.subheader("Explain Individual Prediction")
    transaction_index = st.number_input(
        "Select transaction index",
        min_value=0,
        max_value=len(input_df)-1,
        value=0
    )
    transaction = model_input.iloc[[transaction_index]]
    shap_values_single = shap_explainer(transaction)

    fig, ax = plt.subplots(figsize=(10,4))
    shap.plots.waterfall(shap_values_single[0], show=False)
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

