import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

x=[[1], [5], [10]]
y=[[1], [2], [6]]

linearRegressor = LinearRegression()
reg= linearRegressor.fit(x, y)

#plt.subplot(211)
plt.scatter(x, y, color = 'red')
plt.plot(x, linearRegressor.predict(x), color = 'blue')

print (reg.score(x, y))
print(reg.coef_)
print (np.linalg.norm(y - reg.predict(x)) ** 2)
print (reg)

plt.show()