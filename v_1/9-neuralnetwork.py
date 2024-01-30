import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

train_data = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="data_test_Train")

train_set, test_set = train_test_split(train_data, random_state=42, test_size=0.2)

X_train = train_set[["X", "Y", "Z"]]
y_train = train_set["Cu"]

X_test = test_set[["X", "Y", "Z"]]
y_test = test_set["Cu"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

reg = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
reg.fit(X_train_scaled, y_train)

y_pred = reg.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
print(f'R-squared Score: {r2}')

plt.scatter(y_pred, y_test)
plt.xlabel('Actual Copper Concentration')
plt.ylabel('Predicted Copper Concentration')
plt.title('Actual vs Predicted Copper Concentration (Neural Network Regression)')
plt.plot([min(y_pred), max(y_pred)], [min(y_test), max(y_test)], color='red', label='Regression Line')

plt.legend()
plt.show()
