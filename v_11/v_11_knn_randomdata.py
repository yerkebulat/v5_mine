import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

train_data = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="Original")
sampled_data = train_data.sample(frac=0.0005, random_state=42)
sampled_data['X'] = pd.to_numeric(sampled_data['X'], errors='coerce')
sampled_data['Y'] = pd.to_numeric(sampled_data['Y'], errors='coerce')

X = sampled_data[['X', 'Y']]
y = sampled_data['Cu']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R^2: {r2}")

x_vals = np.linspace(X['X'].min(), X['X'].max(), 1000)
y_vals = np.linspace(X['Y'].min(), X['Y'].max(), 1000)
x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)

points_to_predict = np.c_[x_mesh.ravel(), y_mesh.ravel()]
predicted_values = knn_model.predict(points_to_predict)
predicted_values_mesh = predicted_values.reshape(x_mesh.shape)

cmap = sns.diverging_palette(240, 10, as_cmap=True)

plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(predicted_values_mesh, 
                      cmap=cmap, 
                      vmin=-1.0, 
                      vmax=1.0, 
                      square=True, 
                      cbar_kws={"label": "Cu Concentration"})
plt.title("Random Data - Copper Concentration Heatmap (Predicted values) - KNN Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
