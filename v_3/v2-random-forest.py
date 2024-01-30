import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Read data
train_data = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="data_test_Train")
test_data = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="Data to Predict")
actual_data = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="Original")

# Prepare the features and target variable
X_train = train_data[['X', 'Y']]
y_train = train_data['Cu']

# Initialize the Random Forest Regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust the hyperparameters as needed

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
predicted_values = model.predict(test_data[['X', 'Y']])

# Calculate Mean Squared Error
mse = mean_squared_error(actual_data['Cu'], predicted_values)
print(f"Mean Squared Error: {mse}")

# Plot actual vs predicted values
plt.scatter(actual_data['Cu'], predicted_values)
plt.xlabel('Actual Copper Concentration')
plt.ylabel('Predicted Copper Concentration')
plt.title('Actual vs Predicted Copper Concentration')
plt.show()
