# Loan Eligibility Prediction (Streamlit App)

## Overview

This project presents an end-to-end machine learning pipeline for predicting loan approval using applicant demographic, financial, and property-related attributes. The original notebook-based solution was transformed into a modular, production-oriented Python project with proper logging, error handling, validation, and deployment using Streamlit.

The application enables users to input applicant details and receive a real-time prediction indicating whether the loan is likely to be approved.

Live Application:
https://loan-eligibility-prediction-app-sharmila.streamlit.app/

---

## Problem Statement

Traditional notebook-based machine learning workflows lack modularity, reproducibility, and robustness for real-world deployment. They often fail to handle errors gracefully and do not enforce consistency between training and inference pipelines.

This project addresses these gaps by:
- converting exploratory notebook code into a modular pipeline
- ensuring consistent preprocessing between training and prediction
- incorporating logging and structured error handling
- validating model performance using multiple strategies
- deploying the model for real-time usage

---

## Dataset

The dataset contains structured information about loan applicants and is used to predict loan approval outcomes.

### Features
- Gender  
- Married  
- Dependents  
- Education  
- Self_Employed  
- ApplicantIncome  
- CoapplicantIncome  
- LoanAmount  
- Loan_Amount_Term  
- Credit_History  
- Property_Area  

### Target
- Loan_Approved (binary: 1 = approved, 0 = not approved)

Data preprocessing includes handling missing values, encoding categorical variables, and ensuring numerical consistency across features.

---

## Code Modularization Approach

The project follows a structured and reusable design aligned with real-world ML engineering practices:

- `data_loader.py` → Handles dataset ingestion with error handling  
- `train.py` → Preprocessing, model training, model selection, and artifact saving  
- `evaluate.py` → Computes training and test performance metrics  
- `validation.py` → Performs cross-validation and split-ratio evaluation  
- `predict.py` → Handles inference, preprocessing, and feature alignment  
- `logger.py` → Centralized logging system  
- `custom_exception.py` → Structured exception handling  

A key design decision was to persist feature columns (`columns.pkl`) instead of hardcoding them, ensuring consistency between training and inference and preventing schema mismatch issues.

---

## Modelling Approach

Two classification models were evaluated:

- Logistic Regression  
- Random Forest Classifier  

Logistic Regression was selected as the final model due to its superior generalization performance and stability across validation strategies.

The model pipeline includes:
- StandardScaler for feature scaling  
- Logistic Regression for classification  

Random Forest, while powerful, showed slightly less consistent generalization compared to Logistic Regression in this dataset.

---

## Training Results

Both models were trained on the processed dataset and evaluated on the training set to understand fitting behavior.

Logistic Regression demonstrated balanced learning without excessive overfitting, while Random Forest achieved strong training performance but showed relatively less stable generalization.

---

## Test Results

- Logistic Regression Test Accuracy: 86.18%  

The model achieved strong performance on the hold-out test set, indicating its ability to capture patterns in loan approval data effectively.

---

## Validation Results

To ensure model reliability beyond a single train-test split, additional validation techniques were applied.

### 5-Fold Cross-Validation

Fold Accuracies:
- 82.11%  
- 80.49%  
- 80.49%  
- 75.61%  
- 82.79%  

Average CV Accuracy:
- 80.30%  

Best Fold Accuracy:
- 82.79%  

### Split Ratio Comparison

- 80:20 → 86.18%  
- 75:25 → 86.36%  
- 70:30 → 85.41%  

Best Split Ratio:
- 75:25 with 86.36% accuracy  

### Interpretation

The cross-validation average (80.30%) is lower than the hold-out test accuracy (~86%), indicating that single split results may be slightly optimistic. Cross-validation provides a more conservative and reliable estimate of model generalization.

The relatively small variation across folds suggests that the model is reasonably stable and not highly sensitive to data partitioning.

---

## Streamlit Application

The Streamlit app provides an interactive interface for real-time predictions.

Features:
- User input for applicant attributes  
- Automated preprocessing and feature alignment  
- Real-time loan approval prediction  
- Clean and intuitive interface  

The application uses the saved model and feature schema to ensure consistent predictions.

---

## Logging and Error Handling

The project implements robust logging and exception handling:

- Logs stored in `logs/app.log`  
- Tracks:
  - data loading  
  - preprocessing steps  
  - model training  
  - evaluation and validation  
  - prediction flow  
  - runtime errors with stack traces  

Custom exception handling ensures:
- clear error traceability  
- improved debugging  
- resilient pipeline execution  

---

## Project Structure

```bash
loan-eligibility-prediction-streamlit/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
│
├── data/
│   └── credit.csv
│
├── model/
│   ├── loan_eligibility_model.pkl
│   └── columns.pkl
│
├── logs/
│   └── app.log
│
├── src/
│   ├── config.py
│   ├── custom_exception.py
│   ├── logger.py
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluate.py
│   ├── validation.py
│   └── predict.py
│
└── notebooks/
    └── Loan_Eligibility_Model_Solution.ipynb
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/murisettysharmila28-creator/loan-eligibility-prediction-streamlit
cd loan-eligibility-prediction-streamlit
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

Train the model:

```bash
python main.py
```

Run the Streamlit app:

```bash
python -m streamlit run app.py
```

---

## Key Findings

- Logistic Regression provided better generalization compared to Random Forest  
- Cross-validation revealed that hold-out accuracy can be optimistic  
- Model performance remained relatively stable across different splits  
- Proper preprocessing and feature consistency significantly impact results  

---

## Limitations

- The model is based on historical data and may not reflect real-world lending policies  
- Accuracy is moderate and can be improved with additional features  
- Model assumes input data distribution similar to training data  
- Limited feature engineering was applied  

---

## Challenges

- Handling missing values across multiple categorical and numerical features  
- Ensuring consistent encoding between training and prediction  
- Avoiding schema mismatch during deployment  
- Validating model performance using multiple approaches  

---

## Learning Outcomes

- Transitioned from notebook-based ML to modular project architecture  
- Implemented production-style logging and error handling  
- Applied multiple validation strategies for model reliability  
- Built an end-to-end ML pipeline with deployment  
- Ensured consistency between training and inference  

---

## Future Enhancements

- Hyperparameter tuning for improved performance  
- Integration of advanced models such as Gradient Boosting or XGBoost  
- Probability-based prediction outputs  
- Model explainability using SHAP or feature importance  
- UI improvements for better user experience  

---

## Author

Sharmila Murisetty  
Data Analyst / Business Intelligence Developer