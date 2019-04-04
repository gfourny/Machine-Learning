import matplotlib.pyplot as plt
import numpy as np
import sklearn

def Hypothese(t0,t1,x) :
    return t0 + t1 * x

def AllSumHyp(array, t0, t1, isT1 = False) :
    j = 0
    allHyp = 0
    while(j < len(array)) :
        if isT1 :
            allHyp += (Hypothese(t0, t1, array[j][0]) - array[j][1]) * array[j][0]
        else :
            allHyp += Hypothese(t0, t1, array[j][0]) - array[j][1]
        j += 1
       
    return allHyp

def DisplayGraph(array, x , y) :
    m = 0
    
    plt.figure(1)
    plt.subplot(211)
    plt.scatter(x, y, color = 'red')
    plt.plot(x, Hypothese(t0,t1,x), color = 'blue')

    plt.subplot(212)
    plt.scatter(x, y - Hypothese(t0,t1,x), color = 'red')
    plt.axhline(0, color='blue')

    plt.show()

t0 = 1
t1 = 1
array = [['2019-03-16', 81682], ['2019-03-18', 81720], ['2019-03-20', 81760], ['2019-03-24', 81826], ['2019-03-25', 81844], ['2019-03-26', 81864], ['2019-03-27', 81881], ['2019-03-28', 81900], ['2019-03-30', 81933], ['2019-04-03', 82003]]
i = 1
lastCost = 1

while i < 200000 :
    print("Boucle %f" %i)
    t0inter = t0 - (1/i) / len(array) * AllSumHyp(array,t0,t1)
    t1inter = t1 - (1/i) / len(array) * AllSumHyp(array,t0,t1, True)
    i += 1

    cost = 1 / (2*len(array)) * (AllSumHyp(array, t0, t1)**2)
    print("TETA0 %f" %t0inter)
    print("TETA1 %f" %t1inter)
    print("Cost %f" %round(cost,6))

    if (cost < 0.08) & (round(cost,6) == round(lastCost,6)) :
        i = 200000
    
    lastCost = cost
    t0 = t0inter
    t1 = t1inter

x=np.array([[1], [2], [3]])
y=np.array([[7], [3], [1]])
DisplayGraph(array,x ,y)

# Calcul de regression linéaire sur un compteur gaz