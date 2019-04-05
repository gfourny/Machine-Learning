import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import itertools
import operator
import matplotlib.pyplot as plt

# Tableau de points
A = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [2, 11], [2, 12], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10], [3, 11], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10], [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8], [7, 0], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [8, 0], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [9, 0], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5]]

B = [[1, 15], [2, 14], [2, 15], [3, 13], [3, 14], [3, 15], [4, 12], [4, 13], [4, 14], [4, 15], [5, 11], [5, 12], [5, 13], [5, 14], [5, 15], [6, 10], [6, 11], [6, 12], [6, 13], [6, 14], [6, 15], [7, 9], [7, 10], [7, 11], [7, 12], [7, 13], [7, 14], [7, 15], [8, 8], [8, 9], [8, 10], [8, 11], [8, 12], [8, 13], [8, 14], [8, 15], [9, 7], [9, 8], [9, 9], [9, 10], [9, 11], [9, 12], [9, 13], [9, 14], [9, 15], [10, 6], [10, 7], [10, 8], [10, 9], [10, 10], [10, 11], [10, 12], [10, 13], [10, 14], [10, 15], [11, 5], [11, 6], [11, 7], [11, 8], [11, 9], [11, 10], [11, 11], [11, 12], [11, 13], [11, 14], [11, 15], [12, 4], [12, 5], [12, 6], [12, 7], [12, 8], [12, 9], [12, 10], [12, 11], [12, 12], [12, 13], [12, 14], [12, 15], [13, 3], [13, 4], [13, 5], [13, 6], [13, 7], [13, 8], [13, 9], [13, 10], [13, 11], [13, 12], [13, 13], [13, 14], [13, 15], [14, 2], [14, 3], [14, 4], [14, 5], [14, 6], [14, 7], [14, 8], [14, 9], [14, 10], [14, 11], [14, 12], [14, 13], [14, 14], [14, 15], [15, 1], [15, 2], [15, 3], [15, 4], [15, 5], [15, 6], [15, 7], [15, 8], [15, 9], [15, 10], [15, 11], [15, 12], [15, 13], [15, 14], [15, 15]]

#initialisation d'un graphique de point
plt.plot()
#séparation des coordonnées de chaque point du tableau A
x_a,y_a = [elem[0] for elem in A],[elem[1] for elem in A]
#Affichage des points du tableau A en bleu
plt.scatter(x_a, y_a, color='blue')
#séparation des coordonnées de chaque point du tableau B
x_b,y_b = [elem[0] for elem in B],[elem[1] for elem in B]
#Affichage des points du tableau B en rouge
plt.scatter(x_b, y_b, color='red')
#Concaténation des deux tableaux
data = A + B

target = [0] * len(A) + [1] * len(B)
#classification
classifier = LogisticRegression(solver='lbfgs').fit(data, target)
#demande d'un résultat dans le cas où un point à pour coordonnées [12, 2]
result = classifier.predict([[12,2]])
#Affichage du point de test en vert
plt.scatter(12,2,color='green')

if result == 0 :
    print("La prédiction est A")
else:
    print("La prédiction est B")

plt.show()
