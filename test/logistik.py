import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

x = digits.data
y = digits.target
split = train_test_split(x,y)

# General a toy dataset:s it's just a straight line with some Gaussian noise:
xmin, xmax = -16, 16
n_samples = len(split[0])
X = []
for data1 in split[0] : 
      for data2 in data1 : 
            X.append(int(data2))
X= np.array(X)
y = (X > 0).astype(np.float)

print(X)

X = X[:, np.newaxis]

# Fit the classifier
clf = LogisticRegression(solver='lbfgs')
clf.fit(X, y)

# and plot the result
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.scatter(X.ravel(), y, color='black')
X_test = np.linspace(-5, 10, 300)


def model(x):
    return 1 / (1 + np.exp(-x))


loss = model(X_test * clf.coef_ + clf.intercept_).ravel()
plt.plot(X_test, loss, color='red', linewidth=3)

plt.axhline(.5, color='.5')

plt.ylabel('y')
plt.xlabel('X')
plt.tight_layout()
plt.show()