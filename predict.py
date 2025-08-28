import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Diabetes Prediction - Predict", layout="centered")
st.title("🔮 Predict Diabetes")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 0, 120, 30)
        bmi = st.number_input("BMI", 0.0, 50.0, 25.0)
    with col2:
        hba1c = st.number_input("HbA1c", 0.0, 20.0, 5.0)
        chol = st.number_input("Cholesterol", 0.0, 500.0, 200.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        model = joblib.load("log_model.pkl")
        scaler = joblib.load("scaler.pkl")

        sample = pd.DataFrame([[age, bmi, hba1c, chol]], columns=["AGE","BMI","HbA1c","Chol"])
        sample = scaler.transform(sample)
        pred = model.predict(sample)[0]

        if pred == 1:
            st.error("⚠️ The patient is Diabetic")
        else:
            st.success("✅ The patient is Not Diabetic")
    except:
        st.error("⚠️ Model not found. Please train and save the model first.")
