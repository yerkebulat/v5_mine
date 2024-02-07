import matplotlib.pyplot as plt

models = ['k-nearest neighbors', 'Random Forest', 'XGBoost']
scores = [0.9286561501105778,  0.985537859063075, 0.897371367214794]
scores1 = [0.9214541425967018, 0.9843394252412851, 0.9091167259580043]
scores2 = [0.90031300251671, 0.9809601910757222, 0.9179959891773721]
scores3 = [0.8383134718156167, 0.9731876668739172, 0.9594986772882295]

scores = [score * 100 for score in scores]
scores1 = [score * 100 for score in scores1]
scores2 = [score * 100 for score in scores2]
scores3 = [score * 100 for score in scores3]

plt.figure(figsize=(10, 6))

plt.plot(models, scores, marker='o', linestyle='-', color='blue', label='10000 points')
plt.plot(models, scores1, marker='o', linestyle='-', color='green', label='8000 points')
plt.plot(models, scores2, marker='o', linestyle='-', color='red', label='5000 points')
plt.plot(models, scores3, marker='o', linestyle='-', color='orange', label='2000 points')

plt.title('Correlation coefficient between the reference and estimated maps')
plt.xlabel('Correlation coefficient')
plt.ylabel('Scores, %')
plt.grid(True)
plt.legend()
plt.show()
