# Home Credit Default Risk Project

## Overview

This project builds a machine learning model to predict the probability that a loan applicant will default using the Kaggle Home Credit dataset.

The project includes:
- A reusable data preparation pipeline  
- Feature engineering across multiple relational tables  
- A trained Gradient Boosting model  
- A full model card documenting performance, fairness, explainability, and deployment risks  

---

## Data Preparation

The `data_preparation.py` script performs the following steps:

- Cleans employment anomalies (e.g., unrealistic employment durations)  
- Handles missing values using median imputation  
- Engineers demographic and financial ratio features  
- Aggregates external tables:  
  - Bureau data  
  - Previous applications  
  - Installment payments  
- Removes high-missing columns  
- One-hot encodes categorical variables  
- Aligns training and test datasets  
- Applies final imputation using training data only  

### Output

- `X`: training features  
- `y`: training target  
- `X_test`: test features  

---

## Modeling

### Models Evaluated

1. **Logistic Regression (Baseline)**  
   - Established a simple linear benchmark  

2. **Gradient Boosting (Untuned)**  
   - Strong improvement over baseline  
   - Cross-validated AUC ≈ 0.75  

3. **Gradient Boosting (Tuned)**  
   - Hyperparameters tuned using RandomizedSearchCV  
   - Validation AUC ≈ 0.7245  
   - Kaggle Public Score: **0.71489**  
   - Kaggle Private Score: **0.70177**  

### Final Model

Gradient Boosting was selected due to:
- Strong predictive performance  
- Ability to capture nonlinear relationships  
- Good generalization to unseen data  

---

## Model Card

A full model card is included in:

`model_card.ipynb`

The model card documents:

### Performance Metrics
- ROC-AUC, precision, and recall on validation data  
- Comparison to baseline models  

### Decision Threshold Analysis
- Selection of an optimal probability threshold based on business cost assumptions  
- Trade-offs between false positives and false negatives  

### Explainability (SHAP)
- Feature importance using SHAP values  
- Interpretation of how features influence predictions  
- Translation of model outputs into human-readable explanations  

### Adverse Action Mapping
- Conversion of top predictive features into understandable denial reasons  
- Supports regulatory requirements for explaining credit decisions  

### Fairness Analysis
- Comparison of approval rates across:  
  - Gender (`CODE_GENDER`)  
  - Education level (`NAME_EDUCATION_TYPE`)  
- Identification of disparities between groups  

### Limitations and Risks
- Potential bias in training data  
- Missing or unobserved variables (e.g., informal income, behavioral data)  
- Model sensitivity to economic changes  
- Risk of unfair outcomes across demographic groups  
- Deployment risks including data drift and misuse  

---

## Repository Structure

- `data_preparation.py` — data cleaning and feature engineering pipeline  
- `model_card.ipynb` — full model documentation  
- `Modeling.ipynb` — model development and evaluation  
- `shap_plot.png` — example SHAP visualization  

---

## How to Run

1. Place the required CSV files in the project directory:

- application_train.csv  
- application_test.csv  
- bureau.csv  
- previous_application.csv  
- installments_payments.csv  

2. Run data preparation:

```bash
python data_preparation.py
