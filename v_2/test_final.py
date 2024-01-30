import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score



data = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="data_test_Train")

X = data[['X', 'Y']]
y = data['Cu']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(X_test['X'], X_test['Y'], y_test, color='blue', label='Actual Data', marker='o')
ax.scatter(X_test['X'], X_test['Y'], y_pred, color='red', label='Predicted Data', marker='^')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Copper Concentration')
ax.set_title('Actual vs Predicted Copper Concentrations')

ax.legend()

plt.show()

print(r2_score(y_test, y_pred))
