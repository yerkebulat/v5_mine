import matplotlib.pyplot as plt

models = ['k-nearest neighbors', 'Random Forest', 'XGBoost']
scores = [0.026015297348,  0.006504044119240002, 0.03926404562741593]

scores = [score * 100 for score in scores]

plt.figure(figsize=(10, 6))
plt.plot(models, scores, marker='o', linestyle='-', color='red')
plt.title('Mean squared error (margin error of the models)')
plt.xlabel('Regression Models')
plt.ylabel('Error, %')
plt.grid(True)

for i, txt in enumerate(models):
    plt.annotate(txt, (models[i], scores[i]), textcoords="offset points", xytext=(0, 5), ha='center')

plt.show()
