import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib

scaler=joblib.load('scaler.pkl')
xgb_model=joblib.load('xgb_model.pkl')

st.title("Credit Card Fraud Detection")

st.write("Enter the transaction details to predict if it is fraudulent.")


feature_columns=xgb_model.get_booster().feature_names

exclude_cols=['isolation_forest_prediction', 'lof_prediction']
feature_columns=[col for col in xgb_model.get_booster().feature_names if col not in exclude_cols]


input_data={}
for col in feature_columns:
    input_data[col]= st.text_input(f"Enter value for {col}", value="0.0")


if st.button("Predict"):
    try:
        input_df=pd.DataFrame([input_data])
        input_df=input_df.apply(pd.to_numeric, errors='coerce')

        if input_df.isna().any().any():
            st.error("Invalid input. Please ensure all values are numeric.")
        else:
            input_df=input_df[feature_columns]

            cols_to_scale=['Time', 'Amount']
            if all(col in input_df.columns for col in cols_to_scale):
                input_df[cols_to_scale]= scaler.transform(input_df[cols_to_scale])



            prediction=xgb_model.predict(input_df)
            prediction_proba=xgb_model.predict_proba(input_df)[:,1]

            if prediction[0] ==1:
                st.error(f"This transaction is predicted as Fraudulent (Probability: {prediction_proba[0]:.4f}).")
            else:
                st.success(f"This transaction is predicted as Non-Fraudulent (Probability: {prediction_proba[0]:.4f}).")

    except Exception as e:
        st.error(f"An error occured during prediciton: {e}")