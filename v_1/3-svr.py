import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

train_data = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="data_test_Train")

train_set, test_set = train_test_split(train_data, random_state=42, test_size=0.2)

X_train = train_set[["X", "Y", "Z"]]
y_train = train_set["Cu"]

X_test = test_set[["X", "Y", "Z"]]
y_test = test_set["Cu"]

svr_model = SVR()

svr_model.fit(X_train, y_train)

y_pred = svr_model.predict(X_test)

print(r2_score(y_test, y_pred))

plt.scatter(y_pred, y_test)
plt.xlabel('Actual Copper Concentration')
plt.ylabel('Predicted Copper Concentration')
plt.title('Actual vs Predicted Copper Concentration (Support Vector Regression (SVR) Regression)')
plt.plot([min(y_pred), max(y_pred)], [min(y_test), max(y_test)], color='red', label='Regression Line')

plt.legend()
plt.show()
