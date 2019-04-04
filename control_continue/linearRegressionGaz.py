import numpy as np
from sklearn.linear_model import LinearRegression


X = np.array([[0, 81682], [2, 81720], [4, 81760], [8, 81826], [9, 81844], [10, 81864], [11, 81881], [12, 81900], [14, 81933], [17, 82003]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y)
reg.score(X, y)
reg.coef_
reg.intercept_ 
print(reg.predict(np.array([[3, 5]])))

# Calcul de regression linéaire sur un compteur gaz