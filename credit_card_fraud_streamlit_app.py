#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load deployment objects
deployment_objects = joblib.load("fraud_detection_deployment_objects.pkl")
lr = deployment_objects["logreg"]
rf = deployment_objects["rf"]
xgb_model = deployment_objects["xgb"]
scaler = deployment_objects["scaler"]
shap_xgb = deployment_objects["shap_xgb"]

# App title
st.title("Credit Card Fraud Detection Dashboard")

# Model selection
model_choice = st.selectbox("Select model:", ["Logistic Regression", "Random Forest", "XGBoost"])
model_map = {"Logistic Regression": lr, "Random Forest": rf, "XGBoost": xgb_model}
model = model_map[model_choice]

# Threshold input (manual)
threshold_input = st.text_input("Set prediction threshold (0.0 - 1.0)", value="0.5")
try:
    threshold = float(threshold_input)
    if not 0.0 <= threshold <= 1.0:
        st.error("Threshold must be between 0.0 and 1.0")
        st.stop()
except ValueError:
    st.error("Threshold must be a number")
    st.stop()
st.write(f"Current threshold: {threshold:.10f}")

# Input option: manual or CSV upload
st.subheader("Input transaction data")
input_option = st.radio("Choose input method:", ["Manual Entry", "Upload CSV"])

# Features used during training (exclude 'Hour' initially)
feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# Prepare input DataFrame
if input_option == "Manual Entry":
    input_data = {feature: st.number_input(feature, value=0.0) for feature in feature_names}
    input_df = pd.DataFrame([input_data])
else:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        input_df.columns = input_df.columns.str.strip()
        st.write("Preview of uploaded data:")
        st.dataframe(input_df.head())

        # Check required columns
        missing_cols = [c for c in feature_names if c not in input_df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.stop()

        extra_cols = [c for c in input_df.columns if c not in feature_names]
        if extra_cols:
            st.warning(f"Extra columns will be ignored: {extra_cols}")

        input_df = input_df[feature_names].astype(float)
    else:
        st.stop()

# Add engineered 'Hour' column for display and models expecting it
input_df['Hour'] = (input_df['Time'] // 3600) % 24

# Ensure correct feature order (include Hour)
feature_names_with_hour = feature_names + ['Hour']
input_df = input_df[feature_names_with_hour]

# Prepare model input (exclude 'Hour' for Logistic Regression)
if model_choice == "Logistic Regression":
    # Only original features the LR was trained on
    model_input = input_df[feature_names].copy()
    scaled_input = scaler.transform(model_input)
else:
    # RF/XGB can use Hour
    feature_names_with_hour = feature_names + ['Hour']
    model_input = input_df[feature_names_with_hour].copy()
    scaled_input = model_input.values  # Raw features for RF/XGB

# Make predictions
pred_probs = model.predict_proba(scaled_input)[:, 1]
pred_classes = (pred_probs >= threshold).astype(int)

# Prepare results
results = input_df.copy()
results['Pred_Probability'] = pred_probs
results['Pred_Class'] = pred_classes

st.subheader("Prediction Results")
st.dataframe(results)

# SHAP summary plot for XGBoost only
if model_choice == "XGBoost":
    st.subheader("SHAP Summary Plot (feature importance)")

    # Use only first N rows to speed up
    sample_input = input_df.head(50)  # or min(50, len(input_df))
    shap_values = shap_xgb.shap_values(sample_input)

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values, sample_input, show=False)
    st.pyplot(fig)

# Download predictions
csv = results.to_csv(index=False).encode()
st.download_button(
    label="Download Predictions as CSV",
    data=csv,
    file_name="fraud_predictions.csv",
    mime="text/csv"
)