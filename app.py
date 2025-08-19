import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("xgb_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Churn Prediction App")

st.write("Enter customer details to predict churn probability:")

# Example input fields
tenure = st.number_input("Customer Tenure (months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)
contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Convert categorical input
contract_map = {"Month-to-month":0, "One year":1, "Two year":2}
contract_type = contract_map[contract_type]

# Create input DataFrame
input_data = pd.DataFrame([[tenure, monthly_charges, total_charges, contract_type]],
                          columns=["tenure", "MonthlyCharges", "TotalCharges", "Contract"])

# Scale numerical features
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Churn"):
    prob = model.predict_proba(input_scaled)[0][1]
    st.write(f"Churn Probability: {prob:.2f}")
    if prob > 0.5:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")
