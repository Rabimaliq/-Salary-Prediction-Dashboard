import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set page configurations
st.set_page_config(
    page_title="Salary Prediction Dashboard",
    page_icon="💼",
    layout="wide",
)

# Load the saved model safely
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Error: 'model.pkl' not found.")
    st.stop()

# Load historical data for visualization
@st.cache_data
def load_data():
    try:
        return pd.read_csv('Employers_data.csv')
    except FileNotFoundError:
        try:
            return pd.read_csv('Salary_Data.csv')
        except FileNotFoundError:
            return None

df = load_data()

# --- SIDEBAR INPUTS ---
st.sidebar.header("🔧 Configuration Panel")
experience = st.sidebar.slider(
    "Years of Experience:",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=0.5
)

# --- MAIN PAGE ---
st.title("💼 Salary Prediction Analytics Dashboard")
st.markdown("---")

col1, col2 = st.columns([1, 1.2])  # Adjusted column widths for chart space

with col1:
    st.subheader("🔮 Predictive Analytics Engine")
    input_data = np.array([[experience]])
    prediction = model.predict(input_data)
    
    st.metric(
        label="Estimated Market Value (Annual)",
        value=f"${prediction:,.2f}",
        delta=f"+{experience} Years Exp." if experience > 0 else "Entry Level"
    )
    
    if experience < 2:
        st.info("💡 Profile tier: **Junior Level**.")
    elif experience < 7:
        st.warning("💡 Profile tier: **Mid-Senior Level**.")
    else:
        st.success("💡 Profile tier: **Principal/Lead Level**.")
        
    st.markdown("---")
    st.subheader("📋 Dataset Blueprint")
    if df is not None:
        st.dataframe(df, height=200, use_container_width=True)

with col2:
    st.subheader("📈 Model Regression Line")
    if df is not None:
        # Create standard regression line data points
        x_line = np.linspace(0, 20, 100).reshape(-1, 1)
        y_line = model.predict(x_line)
        line_df = pd.DataFrame({'YearsExperience': x_line.flatten(), 'PredictedSalary': y_line})
        
        # Combine actual dataset points and the line for clean plotting
        chart_data = df.copy()
        chart_data.columns = ['YearsExperience', 'ActualSalary']
        
        # Plot utilizing built-in responsive line charts
        st.line_chart(line_df.set_index('YearsExperience'), y='PredictedSalary', use_container_width=True)
        st.caption("The continuous line tracks your model's prediction trajectory across 0-20 years of experience.")
    else:
        st.info("Upload dataset files to view regression modeling graphs.")
