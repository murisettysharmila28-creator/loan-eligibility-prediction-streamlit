import pickle
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.config import TARGET_COLUMN, FEATURE_COLUMNS, MODEL_DIR, MODEL_PATH
from src.evaluate import evaluate_model
from src.logger import setup_logger

logger = setup_logger()


def preprocess_data(df):
    df = df.copy()

    # Drop ID column
    if "Loan_ID" in df.columns:
        df = df.drop("Loan_ID", axis=1)

    # Fill missing categorical values
    df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
    df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
    df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
    df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
    df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])

    # Fill missing numeric values
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["ApplicantIncome"] = df["ApplicantIncome"].fillna(df["ApplicantIncome"].median())
    df["CoapplicantIncome"] = df["CoapplicantIncome"].fillna(df["CoapplicantIncome"].median())

    # Clean and encode target column
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip().str.upper()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Y": 1, "N": 0})

    # Drop rows where target could not be mapped
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # One-hot encode categorical columns
    df = pd.get_dummies(
        df,
        columns=[
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "Property_Area",
        ],
        dtype=int,
    )

    # Final safeguard
    df = df.fillna(0)

    return df


def prepare_data(df):
    df = preprocess_data(df)

    print("\nTarget dtype:", df[TARGET_COLUMN].dtype)
    print("Target unique values:", df[TARGET_COLUMN].unique())

    x = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    return x, y


def split_data(x, y):
    return train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )


def train_logistic_regression(x_train, y_train):
    model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=3000))
    ])
    model.fit(x_train, y_train)
    logger.info("Logistic Regression model trained successfully.")
    return model


def train_random_forest(x_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)
    logger.info("Random Forest model trained successfully.")
    return model


def save_model(model):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved at: {MODEL_PATH}")


def train_and_select_best_model(df):
    x, y = prepare_data(df)
    joblib.dump(x.columns.tolist(), "model/columns.pkl")
    x_train, x_test, y_train, y_test = split_data(x, y)

    lr_model = train_logistic_regression(x_train, y_train)
    _, lr_test_acc = evaluate_model(lr_model, x_train, y_train, x_test, y_test)

    rf_model = train_random_forest(x_train, y_train)
    _, rf_test_acc = evaluate_model(rf_model, x_train, y_train, x_test, y_test)

    if rf_test_acc > lr_test_acc:
        best_model = rf_model
        best_name = "Random Forest"
        best_score = rf_test_acc
    else:
        best_model = lr_model
        best_name = "Logistic Regression"
        best_score = lr_test_acc

    logger.info(f"Best model selected: {best_name} with Test Accuracy = {best_score:.4f}")
    save_model(best_model)

    return best_model, best_name, best_score


