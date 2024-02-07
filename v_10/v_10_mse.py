import matplotlib.pyplot as plt

models = ['k-nearest neighbors', 'Random Forest', 'XGBoost']
scores = [0.026015297348,  0.006504044119240002, 0.03926404562741593]
scores1 = [0.028799312650000002, 0.0072113483839750005, 0.03583262905619144]
scores2 = [0.036233726439999996, 0.009247466903040001, 0.03339034360558245]
scores3 = [0.05420870244, 0.013924368004450002, 0.01716258986015881]

scores = [score * 100 for score in scores]
scores1 = [score * 100 for score in scores1]
scores2 = [score * 100 for score in scores2]
scores3 = [score * 100 for score in scores3]

plt.figure(figsize=(10, 6))

plt.plot(models, scores, marker='o', linestyle='-', color='blue', label='10000 points')
plt.plot(models, scores1, marker='o', linestyle='-', color='green', label='8000 points')
plt.plot(models, scores2, marker='o', linestyle='-', color='red', label='5000 points')
plt.plot(models, scores3, marker='o', linestyle='-', color='orange', label='2000 points')

plt.title('Mean squared error (margin error of the models)')
plt.xlabel('Regression Models')
plt.ylabel('Error, %')
plt.grid(True)
plt.legend()
plt.show()
