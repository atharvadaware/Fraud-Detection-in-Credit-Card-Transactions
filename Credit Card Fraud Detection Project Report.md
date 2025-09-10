# Credit Card Fraud Detection Project Report

## 1. Introduction 

This project aimed to develop a system for detecting fraudulent credit card transactions. Given the increasing prevalence of online transactions, robust fraud detection is crucial. The project explored both unsupervised anomaly detection techniques and a supervised classification approach to address the challenge of identifying fraudulent transactions from a highly imbalanced dataset.

## 2. Dataset

The project utilized the `creditcard.csv` dataset, which contains anonymized credit card transaction data. The dataset includes numerical features (V1-V28), 'Time', 'Amount', and a binary target variable 'Class' (0 for non-fraudulent, 1 for fraudulent). A key characteristic of this dataset is its significant class imbalance, with a very small percentage of transactions being fraudulent.

## 3. Methodology

The project followed a structured methodology:

-   **Data Loading and Exploration**: The dataset was loaded and initial analysis was performed to understand its structure, identify missing values, and analyze the class distribution.
-   **Data Preprocessing**: Missing values were handled by dropping the single row containing them. The 'Time' and 'Amount' features were scaled using `RobustScaler` to handle outliers effectively.
-   **Data Balancing**: The class imbalance was addressed using the Synthetic Minority Over-sampling Technique (SMOTE) to create a balanced dataset for the supervised learning model.
-   **Anomaly Detection**: Isolation Forest and Local Outlier Factor (LOF) models were applied to the preprocessed data to identify potential anomalies.
-   **Supervised Learning**: An XGBoost classifier was trained on the balanced dataset to classify transactions as fraudulent or non-fraudulent.
-   **Model Evaluation**: The performance of all models was evaluated using relevant metrics, including precision, recall, F1-score, and accuracy. For the XGBoost model, the ROC curve and AUC score were also calculated and plotted.
-   **Web Application Development**: A user interface was developed using Streamlit to allow users to input transaction data and get predictions from the trained XGBoost model.
-   **Deliverables**: The final deliverables included the Jupyter notebook, the web application UI code, and a confusion matrix visualization.

## 4. Models Used

-   **Isolation Forest**: An unsupervised anomaly detection algorithm that isolates outliers based on how easily they are separated from the rest of the data.
-   **Local Outlier Factor (LOF)**: An unsupervised anomaly detection algorithm that measures the local deviation of the density of a given data point with respect to its neighbors.
-   **XGBoost**: A powerful and efficient supervised gradient boosting algorithm widely used for classification and regression tasks.

## 5. Results

-   Initial data exploration confirmed the significant class imbalance.
-   Missing values were successfully handled during preprocessing.
-   SMOTE effectively balanced the training dataset for supervised learning.
-   The anomaly detection models (Isolation Forest and LOF) showed limited performance in accurately identifying fraudulent transactions on their own, as indicated by their classification reports.
-   The XGBoost classifier, trained on the SMOTE-balanced data, achieved exceptional performance on the test set, with perfect precision, recall, and F1-score of 1.00 for both fraudulent and non-fraudulent classes.
-   The ROC curve for the XGBoost model showed an AUC of 1.00, further demonstrating its excellent discriminatory power on the test data.
-   The confusion matrix for the XGBoost model on the test set showed zero false positives and zero false negatives, confirming the perfect classification result on this specific test set.

## 6. Web Application

A Streamlit-based web application (`app.py`) was developed to provide a user-friendly interface for predicting credit card fraud. The application allows users to input transaction details and utilizes the trained XGBoost model and scaler to provide a prediction (Fraudulent or Non-Fraudulent) along with the prediction probability. Instructions for running the application locally were provided.

## 7. Deliverables

The key deliverables for this project are:

-   The Jupyter notebook (`.ipynb` file) containing all the code for data loading, preprocessing, balancing, model training, and evaluation.
-   The Python script for the Streamlit web application (`app.py`).
-   The trained XGBoost model saved as a pickle file (`xgb_model.pkl`).
-   The fitted `RobustScaler` object saved as a pickle file (`scaler.pkl`).
-   A visualization of the confusion matrix for the XGBoost model.

## 8. Conclusion

This project successfully demonstrated the application of machine learning techniques for credit card fraud detection. While anomaly detection models provided some insights, the supervised XGBoost model, trained on a balanced dataset, proved to be highly effective in classifying fraudulent transactions on the test set. The developed Streamlit web application provides a practical way to utilize the trained model for making predictions.

## 9. Future Work

-   Validate the XGBoost model's performance on a larger, independent dataset or real-world transaction data to assess its generalization capabilities.
-   Explore other advanced techniques for handling class imbalance, such as different oversampling or undersampling methods, or using evaluation metrics more suitable for imbalanced datasets (e.g., AUPRC).
-   Investigate the feature importances from the XGBoost model to gain insights into which features are most indicative of fraudulent transactions.
-   Consider deploying the web application to a cloud platform for wider accessibility.
-   Implement continuous monitoring and retraining strategies for the model to adapt to evolving fraud patterns.
