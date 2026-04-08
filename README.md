# Loan Eligibility Prediction App

## Live App  
https://loan-eligibility-prediction-app-sharmila.streamlit.app/

---

## Overview  
This project is a **machine learning web application** that predicts whether a loan application will be approved based on applicant details.

The solution combines:
- Data preprocessing & feature engineering  
- Machine learning model training and evaluation  
- Deployment using Streamlit  

---

## Objective  
To build an end-to-end ML pipeline that:
- Trains classification models on loan data  
- Selects the best-performing model  
- Provides real-time predictions through an interactive web interface  

---

## Model Details  

### Algorithms Used:
- Logistic Regression  
- Random Forest  

### Final Model:
- **Logistic Regression (selected based on performance)**  

### Performance:
- **Test Accuracy:** ~86%  

---

## Features  

- User-friendly input form  
- Real-time loan approval prediction  
- Clean handling of categorical and numerical inputs  
- Automated feature alignment with training data  
- Deployed and accessible online  

---

## Tech Stack  

- **Python**  
- **Pandas / NumPy**  
- **Scikit-learn**  
- **Streamlit**  
- **Joblib**  

---

## Input Parameters  

The app takes the following inputs:

- Gender  
- Marital Status  
- Number of Dependents  
- Education  
- Self Employment Status  
- Applicant Income  
- Coapplicant Income  
- Loan Amount  
- Loan Term (converted to months internally)  
- Credit History (Good / Bad)  
- Property Area  

---

## Data Processing  

- Missing values handled using median/mode imputation  
- Categorical variables encoded using one-hot encoding  
- Target variable (`Loan_Approved`) converted to binary (1/0)  
- Feature alignment ensured between training and inference  

---

## How to Run Locally  

```bash
# Clone repo
git clone <your-repo-link>

# Navigate to project
cd loan-eligibility-prediction-streamlit

# Install dependencies
pip install -r requirements.txt

# Run app
python -m streamlit run app.py
```

```markdown
## Project Structure

The project is organized into modular components for training, inference, and deployment:

loan-eligibility-prediction-streamlit/
│
├── app.py                  # Streamlit application
├── main.py                 # Model training entry point
├── model/
│   ├── loan_eligibility_model.pkl
│   │── columns.pkl
│
├── src/
│   ├── train.py            # Training & preprocessing logic
│   ├── predict.py          # Inference preprocessing
│   └── config.py
│
├── data/
│   └── loan_data.csv
│
└── README.md

## Key Learnings:

- Handling real-world data inconsistencies (missing values, encoding issues)
- Importance of feature alignment between training and prediction
- Model selection and evaluation
- Deploying ML models as interactive applications

## Future Improvements:

- Add probability scores for predictions
- Improve UI/UX design
- Add more advanced models (XGBoost, Gradient Boosting)
- Incorporate model explainability (SHAP / feature importance)

## Author:

Sharmila Murisetty
Graduate Student – Business Intelligence & Systems Infrastructure
Aspiring Data Analyst / BI Developer
```
