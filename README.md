# Credit Card Fraud Detection App

## Project Overview
This project is an interactive **Streamlit dashboard** for detecting credit card fraud using machine learning models.  
It allows users to manually input transaction data or upload CSV files to predict the likelihood of fraud.  

The app supports multiple models:
- **Logistic Regression**  
- **Random Forest**  
- **XGBoost**  

The app also provides **SHAP feature importance** plots for XGBoost to explain predictions and help understand which features contribute most to fraud risk.  

For a live demo, check the Streamlit app here: [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://creditcardfrauddetectionapp1.streamlit.app/) (Streamlit App still under construction).

The project demonstrates:
- Python for **data preprocessing** and ML modeling  
- Pandas & NumPy for **data handling**  
- Matplotlib & SHAP for **visual explanations**  
- Streamlit for **interactive deployment**

---

## Use Case
This app is ideal for:
- **Fraud analysts** detecting suspicious transactions  
- **Data scientists** building explainable ML models  
- **Organizations** monitoring real-time credit card transaction risk  

---

## Features

**Fraud Prediction**
- Predict probability of fraud for individual transactions  
- Choose between **Logistic Regression**, **Random Forest**, or **XGBoost**  
- Set a custom prediction threshold  

**Input Options**
- Manual transaction entry  
- CSV upload for batch predictions  

**Interpretability**
- SHAP summary plots for XGBoost to explain feature importance  

**Data Export**
- Download prediction results as CSV  

---

## Dataset
The Credit Card Fraud Detection dataset was used. [Dataset Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)
The dataset used for model training is based on anonymized credit card transactions containing:
- **Time** (seconds since first transaction)  
- **V1–V28** (anonymized numerical features)  
- **Amount** (transaction amount)  
- **Fraud Label** (0 = Non-Fraud, 1 = Fraud)

The app supports new transaction data in the same feature format.  

---

## Tech Stack

| Category | Tools |
|-----------|-------|
| Programming | Python |
| Data Handling | Pandas, NumPy |
| Machine Learning | scikit-learn, XGBoost, imbalanced-learn |
| Visualization | Matplotlib, Seaborn,SHAP |
| Web App | Streamlit |
| Deployment | Streamlit Cloud |
| Model Persistence | joblib |
| Notebook Analysis | Jupyter Notebook |

---

## Update & Version Log
- **Version 1.0** (March 2026): Initial release with multi-model prediction, CSV upload, manual entry, SHAP visualization for XGBoost, and download functionality
