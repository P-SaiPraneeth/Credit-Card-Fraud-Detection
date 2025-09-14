# Credit-Card-Fraud-Detection
 
 # Project Overview

This project aims to detect fraudulent credit card transactions using machine learning techniques. Fraudulent cases are extremely rare compared to legitimate transactions, making this a highly imbalanced classification problem. The project demonstrates data preprocessing, handling class imbalance, model training, evaluation, and interpretability.
Source: Kaggle – Credit Card Fraud Detection Dataset(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Contains 284,807 transactions, of which only 492 are frauds (0.172%).

# Features
Time, Amount (transaction-specific features)
V1–V28 (anonymized PCA components)
Class (target: 0 = Non-Fraud, 1 = Fraud)

# Workflow
Data Preprocessing
Standardized Amount and Time using StandardScaler.
Train-test split with stratification.
Handling Imbalance
Applied SMOTE (Synthetic Minority Oversampling Technique) to balance classes.
Modeling:
Logistic Regression (baseline model).
Random Forest Classifier (improved performance).
Evaluation Metrics
Confusion Matrix
Classification Report (Precision, Recall, F1-score)
ROC-AUC & Precision-Recall Curve
Feature Importance visualization (Random Forest)

# Results
Random Forest outperformed Logistic Regression in detecting frauds.
Achieved high ROC-AUC (~0.98) and strong precision-recall balance.
Visualizations confirmed that the model effectively separates fraudulent from legitimate transactions.

# Visuals
ROC Curve
Precision-Recall Curve
Feature Importance (Top 15 Features)

# Tech Stack
Languages: Python
Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Imbalanced-learn
ML Models: Logistic Regression, Random Forest

# Future Improvements
Implement XGBoost & LightGBM for better performance.
Hyperparameter tuning with GridSearchCV/RandomizedSearchCV.
Deploy as a web app (Flask/Streamlit).

