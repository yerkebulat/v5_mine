import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor

train_data = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="Original")
sampled_data = train_data.sample(frac=0.0005, random_state=42)
sampled_data['X'] = pd.to_numeric(sampled_data['X'], errors='coerce')
sampled_data['Y'] = pd.to_numeric(sampled_data['Y'], errors='coerce')

X_train = sampled_data[['X', 'Y']]
y_train = sampled_data['Cu']

knn_model = KNeighborsRegressor(n_neighbors=5)  # n_neighbors can be adjusted as needed
knn_model.fit(X_train, y_train)

x_vals = np.linspace(sampled_data['X'].min(), sampled_data['X'].max(), 1000)
y_vals = np.linspace(sampled_data['Y'].min(), sampled_data['Y'].max(), 1000)
x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)

points_to_predict = np.c_[x_mesh.ravel(), y_mesh.ravel()]
predicted_values = knn_model.predict(points_to_predict)
predicted_values_mesh = predicted_values.reshape(1000, 1000)

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
