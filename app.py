import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Set up the web page title
st.title("💼 Salary Prediction App")
st.write("Enter your years of experience below to predict your estimated salary.")

# Create an input field for the user
experience = st.number_input("Years of Experience:", min_value=0.0, max_value=50.0, value=1.0, step=0.5)

# Make a prediction when the user clicks the button
if st.button("Predict Salary"):
    # Reshape input to match the 2D array format expected by the model
    input_data = np.array([[experience]])
    prediction = model.predict(input_data)
    
    # Display the result formatted as currency
    st.success(f"The estimated salary is: ${prediction[0]:,.2f}")
