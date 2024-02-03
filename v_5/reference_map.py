import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

actual_data = pd.read_excel("data_for_Test_and_Train.xlsx", sheet_name="Original")

cmap = sns.diverging_palette(240, 10, as_cmap=True)

plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(actual_data["Cu"].values.reshape(1000, 1000),
                      cmap=cmap,
                      vmin=-1.0,
                      vmax=1.0,
                      square=True,
                      cbar_kws={"label": "Cu Concentration"})
plt.title("Copper Concentration Heatmap (Real-world values)")
plt.show()
