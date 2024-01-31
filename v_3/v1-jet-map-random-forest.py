import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

train_data = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="data_test_Train")
test_data = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="Data to Predict")
actual_data = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="Original")

X_train = train_data[['X', 'Y']]
y_train = train_data['Cu']

model = RandomForestRegressor(n_estimators=100, random_state=42)  

model.fit(X_train, y_train)

predicted_values = model.predict(test_data[['X', 'Y']])

mse = mean_squared_error(actual_data['Cu'], predicted_values)
print(f"Mean Squared Error: {mse}")

cmap = get_cmap('jet')
scatter = plt.scatter(actual_data['Cu'], predicted_values, c=actual_data['Cu'], cmap=cmap)
plt.xlabel('Actual Copper Concentration')
plt.ylabel('Predicted Copper Concentration')
plt.title('Actual vs Predicted Copper Concentration')

cbar = plt.colorbar(scatter)
cbar.set_label('Copper Concentration')

plt.show()
