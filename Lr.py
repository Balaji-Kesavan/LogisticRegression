import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Generate a Balanced Synthetic Dataset
np.random.seed(42)
n_samples = 100

# Generate BMI and Blood Glucose levels for both classes
# High Risk (BMI and Blood Glucose are high)
bmi_high = np.random.uniform(28, 35, n_samples // 2)
blood_glucose_high = np.random.uniform(150, 200, n_samples // 2)
high_risk = np.ones(n_samples // 2)  # Label 1 for high risk

# Low Risk (BMI and Blood Glucose are low)
bmi_low = np.random.uniform(18, 25, n_samples // 2)
blood_glucose_low = np.random.uniform(80, 120, n_samples // 2)
low_risk = np.zeros(n_samples // 2)  # Label 0 for low risk

# Combine data for both classes
bmi = np.concatenate([bmi_high, bmi_low])
blood_glucose = np.concatenate([blood_glucose_high, blood_glucose_low])
risk_class = np.concatenate([high_risk, low_risk])

# Create a DataFrame
data = pd.DataFrame({
    'BMI': bmi,
    'Blood_Glucose_Level': blood_glucose,
    'Risk_Class_Binary': risk_class
})

# Step 2: Split the Data and Train Logistic Regression Model
X = data[['BMI', 'Blood_Glucose_Level']]
y = data['Risk_Class_Binary']

# Standardize the features for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data with stratification
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Get the weights and bias
weights = model.coef_[0]
bias = model.intercept_[0]
print("Weights (w1, w2):", weights)
print("Bias (w0):", bias)

# Step 3: Visualize the Decision Boundary
# Define the decision boundary
x_values = np.linspace(-2, 2, 100)
y_values = -(weights[0] * x_values + bias) / weights[1]  # Solving for y when y(x) = 0

# Plot data points and decision boundary
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='bwr', label='Data Points')
plt.plot(x_values, y_values, color='black', linestyle='--', label='Decision Boundary')
plt.xlabel('Standardized BMI')
plt.ylabel('Standardized Blood Glucose Level')
plt.legend()
plt.title('Diabetes Risk Classification with Linear Decision Boundary')
plt.show()
#test