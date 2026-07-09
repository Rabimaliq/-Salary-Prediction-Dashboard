import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("🔮 Predictive Analytics Engine")
    input_data = np.array([[experience]])
    prediction = model.predict(input_data)
    
    # Extract raw float cleanly
    final_salary = float(prediction[0])
    
    st.metric(
        label="Estimated Market Value (Annual)",
        value=f"${final_salary:,.2f}",
        delta=f"+{experience} Years Exp." if experience > 0 else "Entry Level Base"
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
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # FIX: Select columns by numeric position index (0 and 1) to prevent column name errors
        x_data = df.iloc[:, 0]
        y_data = df.iloc[:, 1]
        
        # Plot actual data points
        ax.scatter(x_data, y_data, color='#FF4B4B', label='Actual Data', alpha=0.7)
        
        # Generate the trendline
        x_line = np.linspace(0, 20, 100).reshape(-1, 1)
        y_line = model.predict(x_line)
        ax.plot(x_line, y_line, color='#1F77B4', linewidth=2, label='Regression Line')
        
        # Mark user slider selection point
        ax.scatter([experience], [final_salary], color='black', s=100, zorder=5, label='Your Selection')
        
        ax.set_xlabel('Years of Experience')
        ax.set_ylabel('Salary')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        
        st.pyplot(fig)
    else:
        st.info("Data files not found. Verify your CSV dataset is inside your repository.")
