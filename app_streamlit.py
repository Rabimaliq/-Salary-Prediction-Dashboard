import streamlit as st
import joblib
import pandas as pd
import os

# 1. Page Configuration
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered", page_icon="🩺")
st.title("🩺 Medical Dashboard: Diabetes Classification")
st.write("Enter the patient's clinical metrics below to evaluate risk using the Decision Tree model.")

# 2. Safely locate and load the model file using an absolute path
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, 'model.pkl') 
    return joblib.load(model_path)
st.write("Current Directory:", os.getcwd())
st.write("Files in directory:", os.listdir("."))
model = load_model()

# 3. Form Layout for Inputs
st.subheader("Patient Attributes")
col1, col2 = st.columns(2)

with col1:
    gender_input = st.selectbox("Gender", options=["Female", "Male"])
    age = st.number_input("Age (years)", min_value=1.0, max_value=120.0, value=45.0, step=1.0)
    bmi = st.number_input("BMI Value", min_value=10.0, max_value=70.0, value=26.5, step=0.1)
    hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=20.0, value=5.5, step=0.1)
    urea = st.number_input("Urea Level", min_value=0.5, max_value=50.0, value=4.5, step=0.1)
    cr = st.number_input("Creatinine (Cr)", min_value=5.0, max_value=1000.0, value=75.0, step=1.0)

with col2:
    chol = st.number_input("Cholesterol (Chol)", min_value=1.0, max_value=20.0, value=4.8, step=0.1)
    tg = st.number_input("Triglycerides (TG)", min_value=0.1, max_value=15.0, value=1.5, step=0.1)
    hdl = st.number_input("HDL Cholesterol", min_value=0.1, max_value=5.0, value=1.2, step=0.1)
    ldl = st.number_input("LDL Cholesterol", min_value=0.1, max_value=15.0, value=2.8, step=0.1)
    vldl = st.number_input("VLDL Cholesterol", min_value=0.1, max_value=15.0, value=0.6, step=0.1)

# Map human-readable gender back to numerical code (1 for Male, 0 for Female)
gender = 1 if gender_input == "Male" else 0

# 4. Process Prediction on Button Click
if st.button("Run Evaluation Result", type="primary"):
    
    # Structure data to match model's expected features exactly
    input_data = pd.DataFrame([{
        'Gender': gender, 'AGE': age, 'Urea': urea, 'Cr': cr, 'HbA1c': hba1c,
        'Chol': chol, 'TG': tg, 'HDL': hdl, 'LDL': ldl, 'VLDL': vldl, 'BMI': bmi
    }])
    
    # Run prediction
    prediction = model.predict(input_data)[0]
    
    st.markdown("---")
    st.subheader("Diagnostic Outcome")
    
    # Stylized card rendering based on decision
    if prediction == 'N':
        st.success("### Result: **Non-Diabetic (N)**  \nPatient exhibits normal glucose regulations.")
    elif prediction == 'P':
        st.warning("### Result: **Pre-Diabetic (P)**  \nElevated risks detected. Lifestyle and preventative intervention recommended.")
    elif prediction == 'Y':
        st.error("### Result: **Diabetic (Y)**  \nMetrics align strongly with diabetic clinical criteria.")
