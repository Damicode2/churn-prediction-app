import streamlit as st
import pandas as pd
import joblib

# Load pipeline
pipeline = joblib.load("churn_pipeline.pkl")

st.title("📊 Customer Churn Prediction App")
st.write("Fill in customer details to predict churn probability")

# Inputs (must match training features)
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])

# Convert to DataFrame
input_data = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract,
    "PaymentMethod": payment_method,
    "InternetService": internet_service,
    "Dependents": dependents,
    # add other categorical features with defaults if needed
}])

# Predict
if st.button("Predict Churn"):
    prob = pipeline.predict_proba(input_data)[0][1]
    st.write(f"**Churn Probability:** {prob:.2f}")
    if prob > 0.5:
        st.error("⚠️ Customer is likely to churn!")
    else:
        st.success("✅ Customer is likely to stay.")
