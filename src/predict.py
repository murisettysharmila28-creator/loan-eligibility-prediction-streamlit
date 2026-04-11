import sys
import pickle
import joblib
import pandas as pd

from src.config import MODEL_PATH, MODEL_DIR
from src.logger import setup_logger
from src.custom_exception import CustomException

logger = setup_logger()


def load_prediction_artifacts():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        columns_path = MODEL_DIR / "columns.pkl"
        feature_columns = joblib.load(columns_path)

        logger.info("Prediction artifacts loaded successfully.")
        return model, feature_columns

    except Exception as e:
        logger.error("Error occurred while loading prediction artifacts.", exc_info=True)
        raise CustomException(e, sys)


def preprocess_input(input_data: dict, feature_columns):
    try:
        input_df = pd.DataFrame([input_data])

        input_df = pd.get_dummies(input_df, dtype=int)

        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        logger.info("Input preprocessing completed successfully.")
        return input_df

    except Exception as e:
        logger.error("Error occurred during input preprocessing.", exc_info=True)
        raise CustomException(e, sys)


def predict_loan_status(input_data: dict):
    try:
        model, feature_columns = load_prediction_artifacts()
        input_df = preprocess_input(input_data, feature_columns)

        prediction = model.predict(input_df)[0]

        logger.info("Loan prediction generated successfully.")
        return int(prediction)

    except Exception as e:
        logger.error("Error occurred during loan prediction.", exc_info=True)
        raise CustomException(e, sys)