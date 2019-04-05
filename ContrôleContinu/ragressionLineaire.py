import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


x=[[0], [2], [4], [8], [9], [10], [11], [12], [14], [17]]
y=[[81682.0], [81720.0], [81760.0], [81826.0], [81844.0], [81864.0], [81881.0], [81900.0], [81933.0], [82003.0]]

#------ Regression Linéaire -------
linearRegressor = LinearRegression()
reg= linearRegressor.fit(x, y)

#affichage du resultat du compteur prédit pour le 18ème jour
print(linearRegressor.predict([[18]]))

#Afficahge des points
print("linear regression done")
plt.subplot(211)
plt.scatter(x, y, color = 'red')
plt.plot(x, linearRegressor.predict(x), color = 'blue')
plt.title("Courbe de regression linéaire")

plt.subplot(212)
plt.scatter(x, y - linearRegressor.predict(x), color = 'red')
plt.axhline(0, color='blue')
plt.title("Courbe des résidus")

#Affichage du graphique
plt.show()