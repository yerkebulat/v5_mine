import matplotlib.pyplot as plt

models = ['k-nearest neighbors', 'Random Forest', 'XGBoost']
scores = [0.8610558611723814,  0.9652627914662635, 0.7902961117986591]

scores = [score * 100 for score in scores]

plt.figure(figsize=(10, 6))
plt.plot(models, scores, marker='o', linestyle='-', color='blue')
plt.title('R-squared score (coefficient of determination)')
plt.xlabel('Regression Models')
plt.ylabel('Scores, %')
plt.grid(True)

for i, txt in enumerate(models):
    plt.annotate(txt, (models[i], scores[i]), textcoords="offset points", xytext=(0, 5), ha='center')

plt.show()
