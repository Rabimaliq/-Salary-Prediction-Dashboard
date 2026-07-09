from logging import info
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# 1. Load data
df = pd.read_csv("Employers_data.csv")

# 2. Separate features and target
x = df[['Experience_Years']]
y = df["Salary"]

# 3. Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42
)

# 4. Initialize and fit the model
model = LinearRegression()
model.fit(x_train, y_train)

# 5. Make predictions on the test set
y_pred = model.predict(x_test)

# 6. EVALUATE

# Print first 5 predictions cleanly
print('First 5 Predicted Values:', np.round(y_pred[:5], 2))

# Extract mathematical parameters
m = model.coef_[0]
b = model.intercept_
print(f"\nModel Formula: Salary = {m:.2f} * Experience + {b:.2f}")

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n--- Model Performance ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} (Average error in thousands)")
print(f"R-squared Score (R2): {r2:.2f} ({r2*100:.1f}% of variance explained)")

# 6. Plotting the results
plt.figure(figsize=(10, 6))

# Plot actual test data points
sns.scatterplot(x=x_test['Experience_Years'], y=y_test, color='blue', label='Actual Data')

# Plot the regression line
plt.plot(x_test['Experience_Years'], y_pred, color='red', linewidth=2, label='Regression Line')

plt.title('Experience vs Salary (Test Set)')
plt.xlabel('Experience_Years')
plt.ylabel('Salary')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Save the trained model to a file named 'model.pkl'
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved successfully as model.pkl")


#streamlit run "C:\Users\rabim\Salary Prediction Model\app.py"
