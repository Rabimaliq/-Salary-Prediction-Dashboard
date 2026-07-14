# 💼 Salary Prediction Model

An end-to-end Machine Learning web application that predicts a professional's salary based on their years of experience. This project uses a **Linear Regression** statistical approach for the predictive backend and implements an interactive frontend interface using **Streamlit**.

## 📈 Model Performance & Mathematical Formula

The model learns the linear relationship between professional experience and financial compensation. Based on the training data, the model generated the following predictive equation:

### 🧮 Regression Equation
$$\text{Salary} = \beta_1 \times \text{Experience\_Years} + \beta_0$$

### 📊 Performance Metrics
The model was validated using a 33% test split ($test\_size=0.33$) and evaluated using standard regression metrics:

* **R-squared ($R^2$) Score:** Evaluates how much variance in salary is explained by years of experience.
* **Root Mean Squared Error (RMSE):** Indicates the average deviation (error margin) between the predicted salary and the actual salary.
* **Mean Squared Error (MSE):** Measures the average squared differences between estimated values and the actual target values.

## 🛠️ Project Structure & Architecture

The workflow is divided into a robust data pipeline, a model training module, and a frontend interface:

1. **Data Pipeline:** Ingests `Employers_data.csv` and separates the independent feature (`Experience_Years`) from the target variable (`Salary`).
2. **Model Training:** Splits data into training/testing subsets using a set `random_state=42` to guarantee reproducibility. Trains a `LinearRegression` model from `scikit-learn`.
3. **Asset Serialization:** Serializes and exports the final fitted model object into a binary format file named `model.pkl` using Python's `pickle` library for low-latency loading.
4. **Interactive Dashboard:** Runs a lightweight web app that loads the frozen model structure to serve instant user predictions.

## 💻 Tech Stack
* **Language:** Python
* **Data Core:** `pandas`, `numpy`
* **Modeling Engine:** `scikit-learn` (Linear Models)
* **Visualization:** `matplotlib`, `seaborn`
* **App Deployment:** `Streamlit`, `pickle`

## 🚀 How to Run Locally

### 1. Install Dependencies
Ensure you have the required packages installed in your local environment:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

### 2. Train and Save the Model
Run your training script to evaluate performance, visualize the regression line, and generate your `model.pkl` asset:
```bash
python train.py
```

### 3. Launch the Web Interface
Deploy your interactive user interface locally using the following Streamlit execution command:
```bash
streamlit run app.py
```
