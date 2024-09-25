# Telco Customer Churn Prediction

This project aims to predict customer churn using various machine learning models, including **Logistic Regression** and **XGBoost**. The goal is to determine which customers are more likely to churn based on various features such as contract type, internet service, monthly charges, and more.

## Project Overview

This project walks through data preprocessing, exploratory data analysis (EDA), feature engineering, model building, and evaluation using multiple machine learning algorithms. Ultimately, it compares **Logistic Regression** and **XGBoost** to understand which performs better in predicting customer churn.

## Data Preprocessing

- **Removed the `customerID` column** as it is irrelevant for modeling.
- **Converted `TotalCharges`** to numeric and handled missing values using coercion.
- **Binary-encoded** categorical columns such as `Partner`, `Dependents`, and `PhoneService`.
- Applied **one-hot encoding** to multi-class features, such as `InternetService`, `Contract`, and `PaymentMethod`.
- Standardized the feature set using **`StandardScaler`** to ensure features are on the same scale, improving model performance.

## Exploratory Data Analysis (EDA)

- **Churn Distribution**: Visualized customer churn by plotting tenure and monthly charges against churn.
- **Key Insights**:
  - Customers with shorter tenure or higher monthly charges are more likely to churn.
  - Histograms showed clear differences in behavior between churned and non-churned customers.

## Feature Engineering

- Replaced `No internet service` and `No phone service` with **`No`** for consistency.
- Converted categorical features like gender and contract type into binary and one-hot encoded formats.

## Modeling

We built and compared two machine learning models:

1. **Logistic Regression**:
   - A linear model used for binary classification.
   - **Accuracy**: 78.8%
   - **Cross-Validation Average Accuracy**: 80.4%
   - Provided interpretable insights through feature coefficients, highlighting key features like **tenure** and **monthly charges**.

2. **XGBoost**:
   - A powerful ensemble boosting method that builds a series of decision trees.
   - **Accuracy**: 79.3%
   - **Cross-Validation Average Accuracy**: 80.6%
   - Captured non-linear relationships between features, with **Contract_Month-to-month**, **InternetService_Fiber optic**, and **tenure** being the most influential features.

## Model Comparison

- **Logistic Regression**:
  - Provided easily interpretable coefficients.
  - **Key Features**:
    - **tenure**: Longer tenure reduces the likelihood of churn.
    - **Monthly Charges**: Higher charges increase churn likelihood.

- **XGBoost**:
  - Outperformed Logistic Regression slightly, with a better ability to capture complex feature interactions.
  - **Key Features**:
    - **Contract_Month-to-month**: Customers with this contract type are more likely to churn.
    - **Internet Service**: Customers with fiber optic or no internet service have higher churn rates.

## Feature Importance

- **Logistic Regression**:
  - Highlighted the impact of individual features through coefficients.
  - **tenure** (negative coefficient) and **monthly charges** (positive coefficient) were the top features.

- **XGBoost**:
  - Visualized feature importance using the **gain** metric.
  - **Contract_Month-to-month** and **InternetService_Fiber optic** were the most important features based on their contribution to model accuracy.

## Conclusion

Both models showed solid performance in predicting customer churn:
- **Logistic Regression** offered simpler and more interpretable results.
- **XGBoost** captured more complex interactions, offering slightly better accuracy.
