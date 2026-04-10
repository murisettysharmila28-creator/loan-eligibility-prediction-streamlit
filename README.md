# Loan Eligibility Prediction

## Live App

https://loan-eligibility-prediction-app-sharmila.streamlit.app/

---

## Overview

This project is an end-to-end machine learning application for predicting whether a loan application is likely to be approved using applicant, financial, and property-related data. The original notebook-based solution was modularized into a reusable Python project and deployed as an interactive Streamlit web application.

The application allows users to enter applicant details and receive a real-time prediction of likely loan approval outcome.

---

## Objective

The goal of this project was to:
- convert notebook-based machine learning code into a modular Python project
- preprocess categorical and numerical loan application data
- compare multiple classification models
- evaluate model performance using classification metrics
- add additional validation for model stability
- deploy the final model through Streamlit

---

## Dataset

The dataset contains applicant, loan, and property-related features used to predict loan approval.

### Features used
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
- Loan_Approved

---

## Models Used

The following classification models were tested:
- Logistic Regression
- Random Forest Classifier

### Selected Model
- **Logistic Regression**

The Logistic Regression model was selected because it achieved the better test performance and generalized well on this dataset.

---

## Evaluation and Validation

### Hold-out Test Result
- **Best Test Accuracy:** 86.18%

### Additional Validation
To assess model stability beyond a single train-test split, additional validation was performed using both 5-fold cross-validation and split ratio comparison.

#### 5-Fold Cross-Validation Accuracy Scores
- 82.11%
- 80.49%
- 80.49%
- 75.61%
- 82.79%

#### Average Cross-Validation Accuracy
- **80.30%**

#### Best Fold Accuracy
- **82.79%**

#### Best-Performing Fold
- **Fold 5**

#### Split Ratio Comparison
- **80:20** → 86.18%
- **75:25** → 86.36%
- **70:30** → 85.41%

#### Best Split Ratio
- **75:25**

#### Best Split Ratio Accuracy
- **86.36%**

### Interpretation
The model performed reasonably well across multiple validation strategies. The 5-fold cross-validation average accuracy of 80.30% provided a more conservative estimate of generalization, while the 75:25 split produced the best hold-out accuracy of 86.36%. This suggests that the Logistic Regression model was stable overall, although single split results were slightly more optimistic than the cross-validation estimate.

---

## Project Features

- Modular code structure
- Separate training, evaluation, validation, and prediction modules
- Missing value handling and preprocessing
- One-hot encoding for categorical features
- Target conversion from Y/N to binary values
- Model comparison using accuracy
- Additional 5-fold cross-validation for model validation
- Split ratio comparison for hold-out evaluation
- Interactive Streamlit interface for loan approval prediction
- Deployed web application for real-time inference

---

## Project Structure

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
    ├── src/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── data_loader.py
    │   ├── train.py
    │   ├── evaluate.py
    │   ├── validate.py
    │   ├── predict.py
    │   └── logger.py
    │
    └── notebooks/
        └── Loan_Eligibility_Model_Solution.ipynb

---

## How to Run Locally

### 1. Clone the repository

    git clone https://github.com/murisettysharmila28-creator/loan-eligibility-prediction-streamlit
    cd loan-eligibility-prediction-streamlit

### 2. Install dependencies

    pip install -r requirements.txt

### 3. Train the model

    python main.py

### 4. Run the Streamlit app

    python -m streamlit run app.py

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## Key Learnings

- Converting notebook code into a modular ML project
- Handling missing values and encoding categorical variables
- Comparing classification models using accuracy
- Using cross-validation and split-ratio comparison to assess model stability
- Saving trained models and feature metadata for deployment
- Building an interactive prediction app using Streamlit

---

## Limitations

- The model is based on the available historical dataset and may not reflect real-world lender-specific approval rules
- The output should be interpreted as a predictive estimate, not an actual lending decision
- Performance may vary if the input data distribution changes significantly

---

## Future Improvements

- Add probability-based interpretation in the user interface
- Perform hyperparameter tuning for the classification models
- Compare additional models such as XGBoost or Gradient Boosting
- Improve validation reporting with more detailed statistical summaries
- Enhance the user interface and explainability of predictions

---

## Author

Sharmila Murisetty - Data Analyst / BI Developer