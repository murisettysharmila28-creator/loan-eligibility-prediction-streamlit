from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Data
DATA_PATH = BASE_DIR / "data" / "credit.csv"

# Model
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "loan_eligibility_model.pkl"

# Target
TARGET_COLUMN = "Loan_Approved"

# Reproducibility
RANDOM_STATE = 42