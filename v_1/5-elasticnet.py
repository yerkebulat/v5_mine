import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
train_data = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="data_test_Train")

# Split data into training and testing sets
train_set, test_set = train_test_split(train_data, random_state=42, test_size=0.2)

# Define features and target variable for training set
X_train = train_set[["X", "Y", "Z"]]
y_train = train_set["Cu"]

# Define features and target variable for testing set
X_test = test_set[["X", "Y", "Z"]]
y_test = test_set["Cu"]

# Create and train the Elastic Net regression model
reg = ElasticNet().fit(X_train, y_train)

# Make predictions on the testing set
y_pred = reg.predict(X_test)

# Evaluate the model
print("R-squared score:", r2_score(y_test, y_pred))

# Plot the predicted vs. actual values
plt.scatter(y_pred, y_test)
plt.xlabel('Actual Copper Concentration')
plt.ylabel('Predicted Copper Concentration')
plt.title('Actual vs Predicted Copper Concentration (Elastic-Net Regression)')
plt.plot([min(y_pred), max(y_pred)], [min(y_test), max(y_test)], color='red', label='Regression Line')

plt.legend()
plt.show()
