import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the scaler and the trained XGBoost model
try:
    scaler = joblib.load('scaler.pkl')
    xgb_model = joblib.load('xgb_model.pkl')
except FileNotFoundError:
    st.error("Scaler or XGBoost model not found. Please ensure 'scaler.pkl' and 'xgb_model.pkl' are in the same directory.")
    st.stop()

st.title("Credit Card Fraud Detection")

st.write("Enter the transaction details to predict if it is fraudulent.")

# Get the list of feature columns from the trained model (excluding 'Class', 'isolation_forest_prediction', and 'lof_prediction')
# Assuming the model was trained on features without 'Class'
feature_columns = [col for col in xgb_model.get_booster().feature_names if col not in ['Class', 'isolation_forest_prediction', 'lof_prediction']]


input_data = {}
for col in feature_columns:
    input_data[col] = st.text_input(f"Enter value for {col}", value="0.0")

if st.button("Predict"):
    try:
        # Convert input data to a pandas DataFrame
        input_df = pd.DataFrame([input_data])

        # Convert all columns to numeric, coercing errors
        input_df = input_df.apply(pd.to_numeric, errors='coerce')

        # Check for any non-numeric inputs after coercion
        if input_df.isnull().any().any():
            st.error("Invalid input. Please ensure all values are numeric.")
        else:
            # Ensure the order of columns in input_df matches the training data
            # Create a DataFrame with all the original feature columns and fill missing ones with 0 or a suitable default
            full_feature_df = pd.DataFrame(columns=xgb_model.get_booster().feature_names)
            full_feature_df.loc[0] = 0 # Or use a more appropriate default value

            # Update the values from the user input for the existing columns
            for col in input_df.columns:
                if col in full_feature_df.columns:
                    full_feature_df[col] = input_df[col]


            # Scale 'Time' and 'Amount' using the fitted scaler
            # Check if 'Time' and 'Amount' are in the columns before scaling
            cols_to_scale = [c for c in ['Time', 'Amount'] if c in full_feature_df.columns]
            if cols_to_scale:
                full_feature_df[cols_to_scale] = scaler.transform(full_feature_df[cols_to_scale])


            # Make prediction
            prediction = xgb_model.predict(full_feature_df)
            prediction_proba = xgb_model.predict_proba(full_feature_df)[:, 1]

            # Display the result
            if prediction[0] == 1:
                st.error(f"This transaction is predicted as Fraudulent (Probability: {prediction_proba[0]:.4f}).")
            else:
                st.success(f"This transaction is predicted as Non-Fraudulent (Probability: {prediction_proba[0]:.4f}).")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")