import matplotlib.pyplot as plt

# Data
models = ['Linear', 'Ridge', 'Support Vector', 'Lasso', 'Elastic Net', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'Neural Network']
scores = [0.05067766203112145, 0.050677908972576624, 0.139172502229719, 0.048963856602964095, 0.05007898324551929, 0.8144681752870332, 0.859247093410204, 0.7773273835675797, 0.08588756779553863]

# Multiply scores by 100
scores = [score * 100 for score in scores]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(models, scores, marker='o', linestyle='-', color='blue')
plt.title('R-squared score (coefficient of determination)')
plt.xlabel('Regression Models')
plt.ylabel('Scores, %')
plt.grid(True)

# Add data labels
for i, txt in enumerate(models):
    plt.annotate(txt, (models[i], scores[i]), textcoords="offset points", xytext=(0, 5), ha='center')

plt.show()
