import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

train_data = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="data_test_Train")
test_data = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="Data to Predict")
actual_data = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="Original")

X_train = train_data[['X', 'Y']]
y_train = train_data['Cu']
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_split_scaled = scaler.fit_transform(X_train_split)
X_val_scaled = scaler.transform(X_val)

mlp_model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', random_state=42)

mlp_model.fit(X_train_split_scaled, y_train_split)

X_test_scaled = scaler.transform(test_data[['X', 'Y']])
predicted_values = mlp_model.predict(X_test_scaled)

mse = mean_squared_error(actual_data['Cu'], predicted_values)
print(f"Mean Squared Error: {mse}")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(actual_data['X'], actual_data['Y'], actual_data['Cu'], c=actual_data['Cu'], cmap='jet', label='Actual Data')
ax.scatter(test_data['X'], test_data['Y'], predicted_values, c=predicted_values, cmap='jet', label='Predicted Data')

ax.set_xlabel('X')
ax.set_ylabel('Y')
#ax.set_zlabel('Copper Concentration')
ax.legend()

desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
image_folder = os.path.join(desktop_path, '3D_Plots')

if not os.path.exists(image_folder):
    os.makedirs(image_folder)

angles = [0, 90, 180, 270, 45, 135, 225, 315]

for angle in angles:
    if angle == 0 or angle == 180:
        ax.view_init(elev=90, azim=angle)
    elif angle == 90 or angle == 270:
        ax.view_init(elev=-90, azim=angle)
    else:
        ax.view_init(elev=20, azim=angle)
    filename = f'nn_3D_Plot_Angle_{angle}.png'
    filepath = os.path.join(image_folder, filename)
    plt.savefig(filepath)
    print(f"Saved {filename}")

plt.show()
