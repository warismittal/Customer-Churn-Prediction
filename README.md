# Customer Churn Prediction

**Churn Analytics: A Machine Learning Approach to Customer Retention**

A machine learning project that predicts whether a customer of a subscription-based digital service is likely to churn. Businesses can use the model to find high-risk customers early and target them with personalized offers, proactive support and engagement campaigns.

## Team

Thapar Institute of Engineering and Technology, Patiala. Submitted to Dr. Ritesh Sharma.

- Waris (1024170155)
- Tanishq (1024170156)
- Kanan (1024170164)
- Samriddhi Gupta (102417065)
- Ananya Dayal (1024170205)


---

## Problem Statement

Customer churn costs subscription businesses revenue, and retaining an existing customer is usually cheaper than acquiring a new one. This project builds a classification model that uses demographics, service usage, subscription duration, billing history and support interactions to flag customers who are likely to leave, and evaluates it with metrics suited to imbalanced data.

## Dataset

- **Source:** Telco Customer Churn dataset (Kaggle)
- **Size:** 7,032 customers, 21 raw features (numeric and categorical)
- **Target:** `Churn` (Yes/No, encoded as 1/0)
- **Class balance:** about 3:1 (5,163 retained vs. 1,869 churned)

## ML Pipeline

1. Data collection
2. Data pre-processing
3. Feature engineering
4. Outlier handling
5. Train–test split
6. Model selection
7. Model training
8. Performance evaluation
9. Comparison and analysis

## Pre-processing and Feature Engineering

- **Cleaning:** converted `TotalCharges` from text to numeric and dropped the 11 zero-tenure rows with missing values instead of imputing them, so the median wouldn't distort the data.
- **Feature selection:** kept 5 core baseline features (e.g., `ChargesPerTenure`, `NumServices`) and removed redundant ones to cut noise.
- **New features:**
  - `EasyLeaver`: paperless billing + electronic check
  - `NoProtection`: no tech-support or security services
- **Encoding and scaling:** one-hot encoding for categorical columns; standardization for tenure, monthly charges and total charges.
- **Class imbalance:** tested SMOTE but dropped it because it added synthetic noise. Used `class_weight='balanced'` instead to improve recall on actual churners.

## Models and Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Random Forest | **0.756** | 0.837 |
| Gradient Boosting | 0.742 | **0.840** |
| XGBoost | 0.741 | 0.839 |
| Logistic Regression | 0.732 | 0.837 |

- **Best accuracy:** Random Forest
- **Best churn detection / class separation:** Gradient Boosting
- **Most stable overall:** Gradient Boosting and XGBoost
- **Most interpretable:** Logistic Regression

All models fall within a 0.73–0.76 accuracy range and share ROC-AUC near 0.84, so the problem is moderately separable and the models discriminate well between churners and non-churners.

## Key Learnings

- Accuracy alone is misleading on imbalanced data. A model that always predicts "no churn" scores about 73%, so recall, F1-score and ROC-AUC are the metrics that matter here.
- Hyperparameter tuning (max depth, learning rate, number of trees) lowered raw accuracy but reduced overfitting, giving more realistic performance on unseen data.

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · Seaborn · Matplotlib


