import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score



train_data = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="data_test_Train")
test_data = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="Data to Predict")
actual_data = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="Original")

X_train = train_data[['X', 'Y']]
y_train = train_data['Cu']

model = LinearRegression()
model.fit(X_train, y_train)

predicted_values = model.predict(test_data[['X', 'Y']])

mse = mean_squared_error(actual_data['Cu'], predicted_values)
print(f"Mean Squared Error: {mse}")

plt.scatter(actual_data['Cu'], predicted_values)
plt.xlabel('Actual Copper Concentration')
plt.ylabel('Predicted Copper Concentration')
plt.title('Actual vs Predicted Copper Concentration')
plt.show()

