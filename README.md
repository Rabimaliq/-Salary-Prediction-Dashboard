# 💼 Salary Prediction Model

An end-to-end Machine Learning web application that predicts a professional's salary based on their years of experience. This project uses a **Linear Regression** statistical approach for the predictive backend and implements an interactive frontend interface using **Streamlit**.

**Live App Link:** [https://salaryexperiencepredict.streamlit.app/]

## 📈 Model Performance & Mathematical Formula

The model learns the linear relationship between Employees Experience in Years and his Salary. 

### 📊 Performance Metrics
The model was validated using a 33% test split ($test\_size=0.33$) and evaluated using standard regression metrics:

* **R-squared ($R^2$) Score:** Evaluates how much variance in salary is explained by years of experience.
* **Root Mean Squared Error (RMSE):** Indicates the average deviation (error margin) between the predicted salary and the actual salary.
* **Mean Squared Error (MSE):** Measures the average squared differences between estimated values and the actual target values.

## 🛠️ Project Structure & Architecture

The workflow is divided into a robust data pipeline, a model training module, and a frontend interface:

1. **Data Pipeline:** Ingests dataset and separates the independent feature (`Experience_Years`) from the target variable (`Salary`).
2. **Model Training:** Splits data into training/testing subsets using a set `random_state=42` to guarantee reproducibility. Trains a `LinearRegression` model from `scikit-learn`.
3. **Asset Serialization:** Serializes and exports the final fitted model object into a binary format file named `model.pkl` using Python's `pickle` library for low-latency loading.
4. **Interactive Dashboard:** Runs a lightweight Streamlit web app that loads the frozen model structure to serve instant user predictions.

## 💻 Tech Stack
* **Language:** Python
* **Data Core:** `pandas`, `numpy`
* **Modeling Engine:** `scikit-learn` (Linear Models)
* **Visualization:** `matplotlib`, `seaborn`
* **App Deployment:** `Streamlit`, `pickle`


