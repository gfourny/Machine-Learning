import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


digits = datasets.load_digits()
print(digits.target)



X = np.array([[1, 0], [0, 1], [3, 2], [2, 3]])
y = np.array([0, 0, 1, 1])
reg = LogisticRegression(solver='lbfgs').fit(X, y)
print(reg.score(X, y))
print(reg.coef_)
print(reg.intercept_)
print(reg.predict(np.array([[3, 5]])))
