import pandas as pd
import numpy as np
import joblib
import streamlit as st

# sklearn + xgboost imports required for unpickling the pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier  # ✅ Required for your saved pipeline

# Load the pipeline
pipeline = joblib.load("churn_pipeline.pkl")

st.title("📊 Customer Churn Prediction App")
st.markdown("Predict whether a customer is likely to churn based on their profile.")

# --- Input fields ---
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=1)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=100.0)

# --- Convert inputs into dataframe ---
input_df = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "InternetService": [internet_service],
    "Contract": [contract],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

# --- Make prediction ---
if st.button("Predict Churn"):
    prediction = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][1]

    st.success(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
    st.info(f"Probability of Churn: {prob:.2f}")
