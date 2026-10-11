# Credit Card Fraud Detection (Two-Stage Cascade Model)

## Overview
A two-stage cascade machine learning model to detect fraudulent credit card transactions. Stage 1 uses Logistic Regression to screen transactions, and Stage 2 uses XGBoost to examine the flagged ones more precisely.

## Dataset
- Source: (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- File name: creditcard.csv.py
- The dataset is not included in this repository. Download it from the link above and place it in the same folder as the code.

## Methodology
- Data loading and preprocessing
- Stage 1: Logistic Regression
- Stage 2: XGBoost
- Model evaluation

## Results
Evaluated on a held-out test set of 56,962 transactions (98 frauds, 0.173% fraud rate).
| Model | Recall | Precision | Frauds Caught | False Alarms |
|-------|--------|-----------|---------------|--------------|
| Single XGBoost | 0.837 | 0.845 | 82 / 98 | 15 |
| Two-Stage Cascade | 0.837 | 0.845 | 82 / 98 | 15 |

## Key findings
- Stage 1 (Logistic Regression, thresholds tuned over 20 random 80/20 splits) cleared 25,739 transactions as genuine with only 1 fraud wrongly cleared.
- Only 31,223 of 56,962 transactions (54.81%) were sent to XGBoost, a **45.19% reduction in XGBoost load**.
- The cascade matched single-XGBoost performance exactly (same recall, precision and false alarms).
- Selected thresholds: T_low = 0.02, T_high = 1.0 (rules passed in 90% of tuning splits).

## How to Run
1. Download the dataset and keep it next to `fraud_detection.py`
2. Install the libraries: `pip install pandas numpy scikit-learn xgboost`
3. Run: `python fraud_detection.py`

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, logistic model

## Author
Tulsi 
