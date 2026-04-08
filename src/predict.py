import pandas as pd

def preprocess_input(input_dict):
    df = pd.DataFrame([input_dict])

    # Drop Loan_ID if exists
    if "Loan_ID" in df.columns:
        df = df.drop("Loan_ID", axis=1)

    # Same encoding as training
    df = pd.get_dummies(df)

    return df

def align_columns(df, training_columns):
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0

    return df[training_columns]