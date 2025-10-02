# Customer Churn Analysis Project

## Overview
This Python project predicts customer churn using a small dataset. It demonstrates **data preprocessing, machine learning model building, evaluation, and visualization**.

## Dataset
- `customer_churn.csv` includes customer purchase history, age, days since last purchase, and churn label.

## Steps
1. Load and explore the data using pandas.
2. Prepare features (`TotalPurchase`, `LastPurchaseDays`, `Age`) and target (`Churned`).
3. Split data into training and testing sets.
4. Train a Random Forest classifier.
5. Evaluate model performance (accuracy, classification report).
6. Visualize feature importance.
7. Optional: Save trained model using joblib.

## How to Run
```bash
pip install pandas scikit-learn matplotlib seaborn joblib
python customer_churn_analysis.py
