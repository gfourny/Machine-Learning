import numpy as np
from sklearn.linear_model import LinearRegression


X = np.array([[['2019-03-16', 81682], ['2019-03-18', 81720], ['2019-03-20', 81760], ['2019-03-24', 81826], ['2019-03-25', 81844], ['2019-03-26', 81864], ['2019-03-27', 81881], ['2019-03-28', 81900], ['2019-03-30', 81933], ['2019-04-03', 82003]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y)
reg.score(X, y)
reg.coef_
reg.intercept_ 
reg.predict(np.array([[3, 5]]))

# Calcul de regression linéaire sur un compteur gaz