# Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions using various machine learning techniques, including anomaly detection models (Isolation Forest and Local Outlier Factor) and a supervised learning model (XGBoost). The project also includes a web application dashboard for making predictions.

## Table of Contents

- Project Overview
- Dataset
- Methodology
- Models Used
- Results
- Web Application
- Deliverables


## Project Overview

The increasing volume of online transactions necessitates robust fraud detection systems. This project tackles the challenge of identifying fraudulent credit card transactions from a highly imbalanced dataset. We explore both unsupervised anomaly detection methods and a supervised classification approach to build a comprehensive fraud detection system.

## Dataset

The dataset used in this project is `creditcard.csv`, which contains anonymized credit card transaction data. It includes numerical features (V1-V28), 'Time', 'Amount', and the target variable 'Class' (0 for non-fraudulent, 1 for fraudulent). The dataset is characterized by a significant class imbalance, with a very small percentage of transactions being fraudulent.

## Methodology

The project follows these main steps:

1.  **Data Loading and Exploration**: Load the dataset and perform initial analysis to understand its structure, identify missing values, and analyze the class distribution.
2.  **Data Preprocessing**: Handle missing values and scale numerical features ('Time' and 'Amount') using `RobustScaler`.
3.  **Data Balancing**: Address the class imbalance issue using the Synthetic Minority Over-sampling Technique (SMOTE).
4.  **Anomaly Detection**: Apply Isolation Forest and Local Outlier Factor models to identify potential anomalies in the dataset.
5.  **Supervised Learning**: Train an XGBoost classifier on the balanced dataset to classify transactions as fraudulent or non-fraudulent.
6.  **Model Evaluation**: Evaluate the performance of all models using relevant metrics, including precision, recall, F1-score, and ROC curve analysis for the XGBoost model.
7.  **Web Application Development**: Create a user interface using Streamlit to allow users to input transaction data and get predictions.
8.  **Deliverables**: Prepare the final deliverables, including the Jupyter notebook, web application UI, and a confusion matrix.

## Models Used

-   **Isolation Forest**: An unsupervised anomaly detection algorithm that isolates outliers.
-   **Local Outlier Factor (LOF)**: An unsupervised anomaly detection algorithm that measures the local deviation density of a data point with respect to its neighbors.
-   **XGBoost**: A supervised gradient boosting algorithm known for its high performance in classification tasks.

## Results

-   The initial data exploration revealed a significant class imbalance.
-   Missing values were successfully handled.
-   SMOTE effectively balanced the dataset for supervised learning.
-   Anomaly detection models (Isolation Forest and LOF) showed limited performance in identifying fraudulent transactions on their own.
-   The XGBoost classifier, trained on the balanced data, achieved excellent performance on the test set, with high precision, recall, and F1-score. The ROC curve analysis also indicated strong discriminatory power.

## Web Application

A Streamlit-based web application is developed to provide an interactive interface for predicting fraudulent transactions. Users can input transaction details, and the application uses the trained XGBoost model to provide a prediction.

## Deliverables

-   Jupyter Notebook containing the complete analysis and model training process.
-   Streamlit web application code (`app.py`).
-   Trained XGBoost model (`xgb_model.pkl`).
-   Fitted `RobustScaler` object (`scaler.pkl`).
-   Confusion matrix plot for the XGBoost model.
