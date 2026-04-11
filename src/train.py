import sys
import pickle
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.config import TARGET_COLUMN, MODEL_DIR, MODEL_PATH
from src.evaluate import evaluate_model
from src.logger import setup_logger
from src.custom_exception import CustomException

logger = setup_logger()


def preprocess_data(df):
    try:
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

        logger.info("Loan dataset preprocessing completed successfully.")
        return df

    except Exception as e:
        logger.error("Error occurred during loan data preprocessing.", exc_info=True)
        raise CustomException(e, sys)


def prepare_data(df):
    try:
        df = preprocess_data(df)

        logger.info(f"Target dtype: {df[TARGET_COLUMN].dtype}")
        logger.info(f"Target unique values: {df[TARGET_COLUMN].unique().tolist()}")

        x = df.drop(TARGET_COLUMN, axis=1)
        y = df[TARGET_COLUMN]

        return x, y

    except Exception as e:
        logger.error("Error occurred while preparing loan data.", exc_info=True)
        raise CustomException(e, sys)


def split_data(x, y):
    try:
        return train_test_split(
            x,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

    except Exception as e:
        logger.error("Error occurred during train-test split.", exc_info=True)
        raise CustomException(e, sys)


def train_logistic_regression(x_train, y_train):
    try:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000))
        ])

        model.fit(x_train, y_train)
        logger.info("Logistic Regression model trained successfully.")
        return model

    except Exception as e:
        logger.error("Error occurred during Logistic Regression training.", exc_info=True)
        raise CustomException(e, sys)


def train_random_forest(x_train, y_train):
    try:
        model = RandomForestClassifier(random_state=42)
        model.fit(x_train, y_train)
        logger.info("Random Forest model trained successfully.")
        return model

    except Exception as e:
        logger.error("Error occurred during Random Forest training.", exc_info=True)
        raise CustomException(e, sys)


def save_model(model):
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

        logger.info(f"Model saved at: {MODEL_PATH}")

    except Exception as e:
        logger.error("Error occurred while saving the selected model.", exc_info=True)
        raise CustomException(e, sys)


def save_feature_columns(columns):
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        columns_path = MODEL_DIR / "columns.pkl"
        joblib.dump(columns, columns_path)

        logger.info(f"Feature columns saved at: {columns_path}")

    except Exception as e:
        logger.error("Error occurred while saving feature columns.", exc_info=True)
        raise CustomException(e, sys)


def train_and_select_best_model(df):
    try:
        x, y = prepare_data(df)
        save_feature_columns(x.columns.tolist())

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

        logger.info(
            f"Best model selected: {best_name} with Test Accuracy = {best_score:.4f}"
        )

        save_model(best_model)

        return best_model, best_name, best_score

    except Exception as e:
        logger.error("Error occurred during model training and selection pipeline.", exc_info=True)
        raise CustomException(e, sys)