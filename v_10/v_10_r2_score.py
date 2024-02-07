import matplotlib.pyplot as plt

models = ['k-nearest neighbors', 'Random Forest', 'XGBoost']
scores = [0.8610558611723814, 0.9652627914662635, 0.7902961117986591]
scores1 = [0.8473761719301216, 0.9617829908205189, 0.8101026547811068]
scores2 = [0.8080778792368595, 0.9510181912242094, 0.8231386559589714]
scores3 = [0.6996370087830981, 0.9228469851450382, 0.9049044416229681]

scores = [score * 100 for score in scores]
scores1 = [score * 100 for score in scores1]
scores2 = [score * 100 for score in scores2]
scores3 = [score * 100 for score in scores3]

plt.figure(figsize=(10, 6))

plt.plot(models, scores, marker='o', linestyle='-', color='blue', label='10000 points')
plt.plot(models, scores1, marker='o', linestyle='-', color='green', label='8000 points')
plt.plot(models, scores2, marker='o', linestyle='-', color='red', label='5000 points')
plt.plot(models, scores3, marker='o', linestyle='-', color='orange', label='2000 points')

plt.title('R-squared score (coefficient of determination)')
plt.xlabel('Regression Models')
plt.ylabel('Scores, %')
plt.grid(True)
plt.legend()
plt.show()
