import matplotlib.pyplot as plt

models = ['Linear', 'Ridge', 'Support Vector', 'Lasso', 'Elastic Net', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'Neural Network']
scores = [0.20895981361765226,  0.24095872665091815, 0.12458327816609025, 0.20926736282989059, 0.20895981361765226, 0.3082785564180001, 0.3072739908010919, 0.2715346306657266, 0.11279087304222068]

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
