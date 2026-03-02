# Home Credit Data Preparation Project

## Overview

This repository contains a reusable Python data preparation script for the Home Credit Default Risk dataset.

The script performs:

- Employment feature cleaning and anomaly handling
- EXT_SOURCE feature cleaning and median imputation
- Demographic and financial feature engineering
- Aggregation of bureau, previous applications, and installment tables
- Removal of high-missing columns
- One-hot encoding of categorical variables
- Alignment of training and test datasets
- Final median imputation using training data only

The output is fully processed modeling-ready datasets:

- X (training features)
- y (training target)
- X_test (test features)

## How to Use

1. Place the following CSV files in the same folder as `data_preparation.py`:

   - application_train.csv  
   - application_test.csv  
   - bureau.csv  
   - previous_application.csv  
   - installments_payments.csv


3. Run the script:

   python data_preparation.py

The script will generate cleaned and aligned datasets ready for modeling.


## Modeling

### Overview

After completing data preparation and feature engineering, multiple classification models were evaluated to predict default risk. Model performance was primarily assessed using ROC-AUC on a validation set and Kaggle public leaderboard score.

---

### Models Evaluated

1. **Baseline Logistic Regression**
   - Used as an initial benchmark model.
   - Provided a reasonable baseline AUC but lacked the flexibility to capture nonlinear relationships.

2. **Gradient Boosting Classifier (Untuned)**
   - Performed significantly better than logistic regression.
   - Cross-validated AUC ≈ 0.75.
   - Demonstrated strong ability to model nonlinear relationships and interactions.

3. **Gradient Boosting Classifier (Hyperparameter Tuned)**
   - RandomizedSearchCV with 3-fold stratified cross-validation.
   - Tuned parameters:
     - `n_estimators`
     - `learning_rate`
     - `max_depth`
     - `min_samples_split`
     - `min_samples_leaf`
   - Validation AUC ≈ 0.7245.
   - Kaggle Public Score: **0.71489**
   - Kaggle Private Score: **0.70177**

Although tuning did not significantly improve performance compared to the untuned model, Gradient Boosting consistently outperformed simpler models and was selected as the final model.

---

### Final Model Selection

Gradient Boosting was selected because:

- It achieved the highest validation AUC among tested models.
- It handled nonlinear relationships and feature interactions effectively.
- It generalized well to the Kaggle test set.
- It produced a competitive public leaderboard score.

---

### Conclusion

The final modeling pipeline integrates engineered financial ratios, demographic features, and aggregated bureau, previous application, and installment payment features. The resulting Gradient Boosting model achieved strong predictive performance and demonstrated the value of feature engineering combined with ensemble learning methods.
