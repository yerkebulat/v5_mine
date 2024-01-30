import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read data
df_train = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="data_test_Train")
df_test = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="Data to Predict")
df_real = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="Original")

# Extract features and target
X_train = df_train[['X', 'Y']]
y_train = df_train['Cu']

# Create Ridge Regression model
alpha = 1.0  # You can adjust the regularization strength (alpha) based on your requirements
model = Ridge(alpha=alpha)

# Train the Ridge Regression model
model.fit(X_train, y_train)

# Predictions on the test and real data
df_test['Predicted_Cu'] = model.predict(df_test[['X', 'Y']])
df_real['Predicted_Cu'] = model.predict(df_real[['X', 'Y']])

# Evaluate the model
mse = mean_squared_error(df_real['Cu'], df_real['Predicted_Cu'])
print(f'Mean Squared Error: {mse}')
r2 = r2_score(df_real['Cu'], df_real['Predicted_Cu'])
print(f'R2 Score: {r2}')

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_real['X'], df_real['Y'], df_real['Cu'], label='Actual Cu', marker='o', s=20)
ax.scatter(df_real['X'], df_real['Y'], df_real['Predicted_Cu'], label='Predicted Cu', marker='x', s=20)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Cu Concentration')
ax.legend()
plt.show()
