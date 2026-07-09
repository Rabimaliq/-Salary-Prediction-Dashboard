import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set page configurations (must be the first Streamlit command)
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
    st.error("Error: 'model.pkl' not found. Please train and save your model first.")
    st.stop()

# Load historical data for visualization if it exists
@st.cache_data
def load_data():
    try:
        return pd.read_csv('Employers_data.csv')
    except FileNotFoundError:
        # Fallback to standard naming convention if file name differs
        try:
            return pd.read_csv('Salary_Data.csv')
        except FileNotFoundError:
            return None

df = load_data()

# --- SIDEBAR INPUTS ---
st.sidebar.header("🔧 Configuration Panel")
st.sidebar.write("Adjust parameters to calculate market predictions.")

experience = st.sidebar.slider(
    "Years of Experience:",
    min_value=0.0,
    max_value=25.0,
    value=5.0,
    step=0.5,
    help="Move the slider to select candidate experience level."
)

# --- MAIN PAGE HEADER ---
st.title("💼 Salary Prediction Analytics Dashboard")
st.markdown("---")

# Layout creation: 2 unequal columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔮 Predictive Analytics Engine")
    
    # Calculate estimation
    input_data = np.array([[experience]])
    prediction = model.predict(input_data)[0]
    
    # Elegant metric display card
    st.metric(
        label="Estimated Market Value (Annual)",
        value=f"${prediction:,.2f}",
        delta=f"+{experience} Years Exp." if experience > 0 else "Entry Level Base"
    )
    
    # User message based on seniority profile
    if experience < 2:
        st.info("💡 Profile tier: **Junior Level**. Focused heavily on core engineering onboarding baselines.")
    elif experience < 7:
        st.warning("💡 Profile tier: **Mid-Senior Level**. Reflects technical execution ownership capacity.")
    else:
        st.success("💡 Profile tier: **Principal/Lead Level**. Reflects strategic decision making architecture scale.")

with col2:
    st.subheader("📋 Dataset Blueprint")
    if df is not None:
        st.write(f"Analyzing internal record matrices ({df.shape[0]} base rows loaded):")
        st.dataframe(df, height=220, use_container_width=True)
    else:
        st.info("No underlying dataset detected in root repository directory.")
