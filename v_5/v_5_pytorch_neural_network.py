import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="data_test_Train")
train_data['X'] = pd.to_numeric(train_data['X'], errors='coerce')
train_data['Y'] = pd.to_numeric(train_data['Y'], errors='coerce')

X_train = train_data[['X', 'Y']].values.astype(np.float32)
y_train = train_data['Cu'].values.astype(np.float32).reshape(-1, 1)

x_vals = np.linspace(train_data['X'].min(), train_data['X'].max(), 1000)
y_vals = np.linspace(train_data['Y'].min(), train_data['Y'].max(), 1000)
x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
points_to_predict = np.c_[x_mesh.ravel(), y_mesh.ravel()]

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1000
for epoch in range(epochs):
    inputs = torch.from_numpy(X_train)
    targets = torch.from_numpy(y_train)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    predicted_values = model(torch.from_numpy(points_to_predict).float()).numpy()

predicted_values_mesh = predicted_values.reshape(1000, 1000)

plt.figure(figsize=(10, 8))
cmap = sns.diverging_palette(240, 10, as_cmap=True)
heatmap = sns.heatmap(predicted_values_mesh,
                      cmap=cmap,
                      vmin=-1.0,
                      vmax=1.0,
                      square=True,
                      cbar_kws={"label": "Cu Concentration"})
plt.title("Copper Concentration Heatmap (Predicted values - PyTorch Neural Network)")
plt.xlabel("X")
plt.ylabel("Y")

plt.show()
