import pandas as pd
import numpy as np

application_train = pd.read_csv("application_train.csv")
application_test  = pd.read_csv("application_test.csv")

bureau = pd.read_csv("bureau.csv")
previous_application = pd.read_csv("previous_application.csv")
installments_payments = pd.read_csv("installments_payments.csv")

def clean_employment_features(df, training_median=None):
    """
    1. Clean employment-related features and flag the anomaly value 365243 in DAYS_EMPLOYED. 
    2. Replace anomaly values with NaN and convert DAYS_EMPLOYED which is currently negative days into positive years. 
    3. Impute missing employment years using the training median.
    
    Params:
        df (DataFrame): Train or test dataframe.
        training_median: Median from training set.
    
    Returns:
        df (DataFrame): Updated dataframe.
        median: Median used for imputation (only returned for training set).
    """

    # Create anomaly indicator column
    df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)

    # Replace anomaly with NaN
    df.loc[df['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = np.nan

    # Convert negative days to positive years
    df['EMPLOYMENT_YEARS'] = -df['DAYS_EMPLOYED'] / 365

    # If median not supplied, compute from this dataset (training phase)
    if training_median is None:
        training_median = df['EMPLOYMENT_YEARS'].median()

    # Impute missing employment years
    df['EMPLOYMENT_YEARS'] = df['EMPLOYMENT_YEARS'].fillna(training_median)

    return df, training_median

# Apply to the training set first to compute the median 
application_train, employment_median = clean_employment_features(application_train)

# Apply to the test set using the median that was just computed using the training set 
application_test, _ = clean_employment_features(application_test, employment_median)



def clean_ext_source_features(df, training_medians=None):
    """
    Cleans EXT_SOURCE features.
    1. Creates missing indicator columns for EXT_SOURCE_1, 2, 3
    2. Computes medians
    3. Imputes missing values using training medians
    4. Creates overall_EXT_SOURCE as row-wise mean

    Params:
        df (DataFrame): Train or test dataframe.
        training_medians: Dictionary of medians computed from training set.

    Returns:
        df (DataFrame): Updated dataframe.
        training_medians: Medians used for imputation.
    """

    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

    # Define the missing indicator columns
    for col in ext_cols:
        df[col + '_MISSING'] = df[col].isna().astype(int)

    # Compute the medians
    if training_medians is None:
        training_medians = {
            col: df[col].median() for col in ext_cols
        }

    # Impute Nans with medians
    for col in ext_cols:
        df[col] = df[col].fillna(training_medians[col])

    # Create the new overall_EXT_SOURCE column using row-wise means
    df['overall_EXT_SOURCE'] = df[ext_cols].mean(axis=1)

    return df, training_medians

# Apply to the training data set, compute the medians
application_train, ext_medians = clean_ext_source_features(application_train)

# Apply to the test set using above medians
application_test, _ = clean_ext_source_features(application_test, ext_medians)



def engineer_demographic_features(df, drop_original=False):
    """
    Engineer demographic features related to age.

    1. Converts DAYS_BIRTH (negative days) into positive AGE_YEARS.
    2. Optionally drops original DAYS_BIRTH column.

    Params:
        df (DataFrame): Train/test dataset.
        drop_original (bool): Whether to drop raw day columns.

    Returns:
        df (DataFrame): Updated dataset.
    """

    # Convert negative days to positive age in years
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365

    # Optional to drop the original column if user specifies
    if drop_original:
        df.drop(columns=['DAYS_BIRTH'], inplace=True)

    return df

application_train = engineer_demographic_features(application_train)
application_test  = engineer_demographic_features(application_test)


def engineer_financial_ratios(df):
    """
    Engineer financial ratio features to enhance the credit risk modeling abilities. 

    Ratios to be added:CREDIT_TO_INCOME, ANNUITY_TO_INCOME, LOAN_TO_VALUE, INCOME_PER_FAMILY_MEMBER

    Params:
        df (DataFrame): Train/test dataset.

    Returns:
        df (DataFrame): Updated dataset.
    """

    # Avoid division by zero
    eps = 1e-6

    # Calculate the credit to income ratio
    df['CREDIT_TO_INCOME'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + eps)

    # Calculate Loan annuity relative to income
    df['ANNUITY_TO_INCOME'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + eps)

    # Calculate loan to value ratio
    df['LOAN_TO_VALUE'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + eps)

    # Calculate loan to income 
    df['INCOME_PER_FAMILY_MEMBER'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + eps)

    return df

# Apply to the training and test data 
application_train = engineer_financial_ratios(application_train)
application_test  = engineer_financial_ratios(application_test)


def drop_high_proportion_missing_columns(train_df, test_df, threshold=0.60):
    """
    Drops any column with missing proportion above threshold.

    Params:
        train_df (DataFrame): Training dataset.
        test_df (DataFrame): Test dataset.
        threshold : Proportion of missing values required to drop column.

    Returns:
        train_df (DataFrame): Updated training dataset.
        test_df (DataFrame): Updated test dataset.
        dropped_cols : List of dropped column names.
    """

    # Compute the proportion of missing values for each column within the dataframe
    missing_proportions = train_df.isna().mean()

    # Identify the proportions which exceed the threshold
    dropped_cols = missing_proportions[missing_proportions > threshold].index.tolist()

    # Drop the high missing columns from both datasets
    train_df = train_df.drop(columns=dropped_cols)
    test_df = test_df.drop(columns=dropped_cols)

    return train_df, test_df, dropped_cols

# Apply to the train and test sets 
application_train, application_test, dropped_columns = drop_high_proportion_missing_columns(application_train, application_test, threshold=0.60)


# Start to handle the categorical variables 
application_train = pd.get_dummies(application_train)
application_test  = pd.get_dummies(application_test)

# Align columns so both datasets match
application_train, application_test = application_train.align(application_test,join='left',axis=1,fill_value=0)

def final_imputation(train_df, test_df):
    """
    Performs final imputation:
    
    1. Numeric columns: median (using the training set only)
    2. Categorical columns: mode (using the training set only)

    Returns:
        train_df, test_df
    """

    # First separate numeric and categorical columns
    numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = train_df.select_dtypes(include=['object']).columns

    # Do numeric imputation using the median
    numeric_medians = train_df[numeric_cols].median()

    train_df[numeric_cols] = train_df[numeric_cols].fillna(numeric_medians)
    test_df[numeric_cols] = test_df[numeric_cols].fillna(numeric_medians)

    # Do cateogorical imputation using modes
    if len(categorical_cols) > 0:
      categorical_modes = train_df[categorical_cols].mode().iloc[0]
      train_df[categorical_cols] = train_df[categorical_cols].fillna(categorical_modes)
      test_df[categorical_cols] = test_df[categorical_cols].fillna(categorical_modes)

    return train_df, test_df

# Apply the final imputation to get rid of all missing values in numeric and categorical columns
application_train, application_test = final_imputation(application_train,application_test)

# Separate target variable from the training data set 
y = application_train['TARGET']

# Drop target from training features
X = application_train.drop(columns=['TARGET'])

# Test set does NOT contain TARGET
X_test = application_test.copy()


def aggregate_bureau(bureau_df):
    """
    Aggregate bureau data to applicant level.
    """

    # Calculate the total number of previous credits
    bureau_agg = bureau_df.groupby('SK_ID_CURR').agg({
        'SK_ID_BUREAU': 'count',
        'AMT_CREDIT_SUM_DEBT': 'mean',
        'CREDIT_ACTIVE': lambda x: (x == 'Active').sum()
    }).rename(columns={
        'SK_ID_BUREAU': 'BUREAU_COUNT',
        'AMT_CREDIT_SUM_DEBT': 'BUREAU_MEAN_DEBT',
        'CREDIT_ACTIVE': 'BUREAU_ACTIVE_COUNT'
    })

    bureau_agg.reset_index(inplace=True)

    return bureau_agg


def aggregate_previous_applications(prev_df):
    """
    Aggregate previous applications to applicant level.
    """

    prev_df['APPROVED_FLAG'] = (prev_df['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)

    prev_agg = prev_df.groupby('SK_ID_CURR').agg({
        'SK_ID_PREV': 'count',
        'APPROVED_FLAG': 'mean'
    }).rename(columns={
        'SK_ID_PREV': 'PREV_APP_COUNT',
        'APPROVED_FLAG': 'PREV_APPROVAL_RATE'
    })

    prev_agg.reset_index(inplace=True)

    return prev_agg

def aggregate_installments(inst_df):
    """
    Aggregate installment payment behavior.
    """

    # Calculate delay, positive numbers mean that it is late
    inst_df['PAYMENT_DELAY'] = inst_df['DAYS_ENTRY_PAYMENT'] - inst_df['DAYS_INSTALMENT']

    inst_df['LATE_PAYMENT_FLAG'] = (inst_df['PAYMENT_DELAY'] > 0).astype(int)

    inst_agg = inst_df.groupby('SK_ID_CURR').agg({
        'PAYMENT_DELAY': 'mean',
        'LATE_PAYMENT_FLAG': 'mean'
    }).rename(columns={
        'PAYMENT_DELAY': 'MEAN_PAYMENT_DELAY',
        'LATE_PAYMENT_FLAG': 'LATE_PAYMENT_RATE'
    })

    inst_agg.reset_index(inplace=True)

    return inst_agg


# Create aggregated stables
bureau_agg = aggregate_bureau(bureau)
prev_agg = aggregate_previous_applications(previous_application)
inst_agg = aggregate_installments(installments_payments)

# Merge bureau aggregates
application_train = application_train.merge(bureau_agg, on='SK_ID_CURR', how='left')
application_test  = application_test.merge(bureau_agg, on='SK_ID_CURR', how='left')

# Merge previous application aggregates
application_train = application_train.merge(prev_agg, on='SK_ID_CURR', how='left')
application_test  = application_test.merge(prev_agg, on='SK_ID_CURR', how='left')

# Merge installment aggregates
application_train = application_train.merge(inst_agg, on='SK_ID_CURR', how='left')
application_test  = application_test.merge(inst_agg, on='SK_ID_CURR', how='left')


# Seperate the target variable before encoding
y = application_train['TARGET']
application_train = application_train.drop(columns=['TARGET'])


# One-Hot encode the cateogorical variables

application_train = pd.get_dummies(application_train)
application_test  = pd.get_dummies(application_test)


# Align the train and test columns 

application_train, application_test = application_train.align(
    application_test,
    join='left',
    axis=1,
    fill_value=np.nan
)


# Final imputation using training medians
application_train, application_test = final_imputation(
    application_train,
    application_test
)


X = application_train.copy()
X_test = application_test.copy()

print("Final training shape:", X.shape)
print("Final test shape:", X_test.shape)
print("Missing values remaining (train):", X.isna().sum().sum())
print("Missing values remaining (test):", X_test.isna().sum().sum())
