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

# --- MAIN PAGE ---
st.title("💼 Salary Prediction Analytics Dashboard")
st.markdown("---")

col1, col2 = st.columns([1, 1.2])

with col1:
    # ADDED: Your clean text-based numeric model input interface block
    st.subheader("Model Inputs")
    exp_input = st.number_input(
        "Years of Experience:",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5
    )
    
    st.markdown("---")
    
    # Calculate estimation using your new 'exp_input' variable name
    input_data = np.array([[exp_input]])
    prediction = model.predict(input_data)
    
    # Extract raw float cleanly
    if hasattr(prediction, "item"):
        final_salary = float(prediction.item())
    else:
        final_salary = float(np.array(prediction).ravel())
    
    st.metric(
        label="Estimated Market Value (Annual)",
        value=f"${final_salary:,.2f}",
        delta=f"+{exp_input} Years Exp." if exp_input > 0 else "Entry Level Base"
    )
    
    if exp_input < 2:
        st.info("💡 Profile tier: **Junior Level**.")
    elif exp_input < 7:
        st.warning("💡 Profile tier: **Mid-Senior Level**.")
    else:
        st.success("💡 Profile tier: **Principal/Lead Level**.")

with col2:
    st.subheader("📈 Model Regression Line")
    
    # Generate the prediction trendline values dynamically from the model
    x_range = np.linspace(0, 50, 101)  # Expanded to 50 to match your input ceiling boundary
    predictions = [float(model.predict(np.array([[x]])).item()) for x in x_range]
    
    # Structure the line chart DataFrame
    chart_df = pd.DataFrame({
        "Years of Experience": x_range,
        "Predicted Salary ($)": predictions
    }).set_index("Years of Experience")
    
    # Render interactive line graph onto the dashboard canvas
    st.line_chart(chart_df, color="#1F77B4", use_container_width=True)
    st.caption("The trendline tracks your machine learning model's path across 0-50 years of experience.")
