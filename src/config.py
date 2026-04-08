from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "credit.csv"

MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "loan_eligibility_model.pkl"

TARGET_COLUMN = "Loan_Approved"

FEATURE_COLUMNS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Credit_History",
    "Loan_Amount_Term",
    "Gender_Female",
    "Gender_Male",
    "Married_No",
    "Married_Yes",
    "Dependents_0",
    "Dependents_1",
    "Dependents_2",
    "Dependents_3+",
    "Education_Graduate",
    "Education_Not Graduate",
    "Self_Employed_No",
    "Self_Employed_Yes",
    "Property_Area_Rural",
    "Property_Area_Semiurban",
    "Property_Area_Urban",
]