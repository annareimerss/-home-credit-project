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

2. Run the script:

   python data_preparation.py

The script will generate cleaned and aligned datasets ready for modeling.
