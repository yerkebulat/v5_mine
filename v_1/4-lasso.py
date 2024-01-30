import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
train_data = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="data_test_Train")

# Split the data into training and test sets
train_set, test_set = train_test_split(train_data, random_state=42, test_size=0.2)

# Define features (X) and target variable (y) for training set
X_train = train_set[["X", "Y", "Z"]]
y_train = train_set["Cu"]

# Define features (X) and target variable (y) for test set
X_test = test_set[["X", "Y", "Z"]]
y_test = test_set["Cu"]

# Create and train the Lasso Regression model
lasso_reg = Lasso(alpha=1.0)  # You can adjust the alpha parameter based on your needs
lasso_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lasso_reg.predict(X_test)

# Evaluate the model
print("R2 Score:", r2_score(y_test, y_pred))

# Plot the results
plt.scatter(y_pred, y_test)
plt.xlabel('Actual Copper Concentration')
plt.ylabel('Predicted Copper Concentration')
plt.title('Actual vs Predicted Copper Concentration (Lasso Regression)')
plt.plot([min(y_pred), max(y_pred)], [min(y_test), max(y_test)], color='red', label='Regression Line')
plt.legend()
plt.show()
