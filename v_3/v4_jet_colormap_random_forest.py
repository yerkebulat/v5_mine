import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

train_data = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="data_test_Train")
test_data = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="Data to Predict")
actual_data = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="Original")

X_train = train_data[['X', 'Y']]
y_train = train_data['Cu']

X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_split, y_train_split)

predicted_values = rf_model.predict(test_data[['X', 'Y']])

actual_values = actual_data['Cu']

mse = mean_squared_error(actual_values, predicted_values)
print(f"Mean Squared Error: {mse}")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(actual_data['X'], actual_data['Y'], actual_values, c=actual_values, cmap='jet', label='Actual Data')

ax.scatter(test_data['X'], test_data['Y'], predicted_values, c=predicted_values, cmap='jet', label='Predicted Data')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Copper Concentration')
ax.legend()

plt.show()
