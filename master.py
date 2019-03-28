import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

df = pd.read_csv('master.csv')
year = df.groupby('year').size()
print(df.groupby('year').size())

year_column = df.iloc[:,[1]].as_matrix()
print(year_column)

year.hist()
plt.show()
#year = df.groupby('year').mean()

#year_column = df.iloc[:,[1]].as_matrix()
#print(year_column)

#suicide_column = df.iloc[:,[4]].as_matrix()
#print(suicide_column)

#linearRegressor = LinearRegression()
#reg = linearRegressor.fit(year_column, suicide_column)

#plt.scatter(year_column, suicide_column, color = 'red')
#plt.plot(year, linearRegressor.predict(year_column), color = 'blue')

#plt.show()