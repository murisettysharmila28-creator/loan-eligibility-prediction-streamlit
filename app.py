import streamlit as st
import joblib
import pandas as pd
from src.predict import preprocess_input, align_columns

st.set_page_config(page_title="Loan Eligibility Predictor", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("model/loan_eligibility_model.pkl")

@st.cache_resource
def load_columns():
    return joblib.load("model/columns.pkl")

model = load_model()
columns = load_columns()

st.title("Loan Eligibility Prediction App")
st.write("Enter applicant details to predict whether the loan is likely to be approved.")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Marital Status", ["Yes", "No"])
dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

app_income = st.number_input("Applicant Income", min_value=0.0, value=5000.0, step=100.0)
coapp_income = st.number_input("Coapplicant Income", min_value=0.0, value=1500.0, step=100.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0, value=128.0, step=1.0)

loan_term_label = st.selectbox(
    "Loan Term",
    ["5 years", "10 years", "15 years", "20 years", "30 years"]
)

loan_term_mapping = {
    "5 years": 60,
    "10 years": 120,
    "15 years": 180,
    "20 years": 240,
    "30 years": 360,
}
loan_term = loan_term_mapping[loan_term_label]

credit_history_label = st.selectbox("Credit History", ["Good", "Bad"])
credit_history = 1.0 if credit_history_label == "Good" else 0.0

property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.button("Predict"):
    input_dict = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": app_income,
        "CoapplicantIncome": coapp_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
    }

    input_df = preprocess_input(input_dict)
    input_df = align_columns(input_df, columns)

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Not Approved")