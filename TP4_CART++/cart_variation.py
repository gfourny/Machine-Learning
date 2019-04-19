from sklearn.datasets import fetch_covtype
rcv1 = fetch_covtype()

from sklearn.datasets import fetch_covtype
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import itertools
import operator
import time

#Awfull Hack to mute warning about 
# convergence issues
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def generateOvRClassifier(classes, x_train,y_train):
    o_vs_r_classifiers = {}
    for elem in classes:
        class_valid = [x_train[index] for index, value in enumerate(y_train) if value == elem]
        class_invalid = [x_train[index] for index, value in enumerate(y_train) if value != elem]
        value = [1] * len(class_valid) + [0] * len(class_invalid)
        learn = class_valid + class_invalid
        o_vs_r_classifiers["%d_rest" % elem] = LogisticRegression(multi_class='ovr',solver='lbfgs').fit(learn, value)
    return o_vs_r_classifiers


def predictOVR(test_values, o_vs_r_classifiers):
    results = {}
    i=0
    for elem in test_values:
        intern_result = {}
        for name, classifier in o_vs_r_classifiers.items():
            result = classifier.predict([elem[0]])
            result_proba = classifier.predict_proba([elem[0]])
            intern_result[name.split('_')[0]] = result_proba[0][1]
        results[i] = intern_result
        i+=1
    correct = 0
    for key, elem in results.items():
        predicted = max(elem.items(), key=operator.itemgetter(1))[0]
        value = test_values[key][1]
        if int(predicted) == value:
            correct +=1
        #print("Predicted %s and value was %s" %(predicted,value))
    prct = (correct/len(results)*100)
    print(f"The One versus Rest score a {prct} % precision score ")

def generateOvOClassifier(classes, x_train,y_train):
    o_vs_o_classifiers = {}
    for elem in itertools.combinations(classes,2):
        class0 = [x_train[index] for index, value in enumerate(y_train) if value == elem[0]]
        class1 = [x_train[index] for index, value in enumerate(y_train) if value == elem[1]]
        value = [0] * len(class0) + [1] * len(class1)
        learn = class0 + class1
        o_vs_o_classifiers['%d_%d'%elem] = LogisticRegression(solver='lbfgs').fit(learn, value)
    return o_vs_o_classifiers

def predictOVO(test_values, o_vs_o_classifiers):
    """
    TO DO : STATS
    """
    results = {}
    i=0
    for elem in test_values:
        intern_result = {}
        for name,classifiers in o_vs_o_classifiers.items():
            result = classifiers.predict([elem[0]])
            members = name.split('_')
            if intern_result.get(members[result[0]]):
                intern_result[members[result[0]]] += 1
            else:
                intern_result[members[result[0]]] = 1
        results[i] = intern_result
        i+=1
    correct = 0
    for key,elem in results.items():
        predicted = max(elem.items(), key=operator.itemgetter(1))[0]
        value = test_values[key][1]
        if int(predicted) == value:
            correct += 1
        #print("Predicted %s and value was %s" %(predicted,value))
    prct = (correct/len(results)*100)
    print(f"The One versus One score a {prct} % precision score ")

def generateForetClassifier(classes, x_train,y_train):
    return RandomForestClassifier(n_estimators=10).fit(x_train, y_train)

def predictForet(test_values, foret_classifiers):
    correct = 0
    for elem in test_values:
        result = foret_classifiers.predict([elem[0]])
        if (elem[1]==result):
            correct +=1

    prct = (correct/len(test_values)*100)
    print(f"Le Forest score a {prct} % de precision")


def generateSVMClassifier(classes, x_train,y_train) : 
    return svm.SVC(gamma='scale', decision_function_shape='ovo', probability=True).fit(x_train, y_train)

def predictSVM(test_values, SVM_classifiers):
    correct = 0
    for elem in test_values:
        result = SVM_classifiers.predict([elem[0]])
        if (elem[1]==result):
            correct +=1

    prct = (correct/len(test_values)*100)
    print(f"Le SVM score a {prct} % de precision")

def classifieur0v0(classes, test_values,x_train,y_train):
    startime0 = time.time()
    #Creation du classifieur O v O 
    print("Creation du classifieur OVO- Temps de realisation : ")
    starttime1 = time.time()
    o_vs_o_classifiers = generateOvOClassifier(classes, x_train,y_train)
    time1= time.time()-starttime1
    
    print(time1)
    # Lancement de la prédiction 
    print("Prediction du classifieur OVO- Temps de realisation : ")
    starttime2 = time.time()
    predictOVO(test_values, o_vs_o_classifiers)
    time2= time.time()-starttime2
    print(time2)

    print("Realisation totale du processus 0V0 - Temps de réalisation :")
    totalTime = time.time() - startime0
    print(totalTime)

def classifieur0vR(classes, test_values,x_train,y_train) :
    startime0 = time.time()
    print("Creation du classifieur OVR- Temps de realisation : ")
    starttime = time.time()
    ovrclassifier = generateOvRClassifier(classes, x_train,y_train)
    time1= time.time()-starttime
    print((time1))

    print("Prediction du classifieur OVR- Temps de realisation : ")
    starttime = time.time()
    predictOVR(test_values,ovrclassifier)
    time2= time.time()-starttime
    print(time2)
    

    print("Realisation totale du processus 0VR - Temps de réalisation :")
    totalTime = time.time() - startime0
    print(totalTime)

def foretClassifieur(classes, test_values,x_train,y_train) : 
    startime0 = time.time()
    print("Creation du classifieur Foret - Temps de réalisation :")
    starttime = time.time()
    foret_classifiers = generateForetClassifier(classes, x_train,y_train)
    time1= time.time()-starttime
    print(time1)

    print("Prediction du classifieur Foret- Temps de realisation : ")
    starttime = time.time()
    predictForet(test_values, foret_classifiers)
    time2= time.time()-starttime
    print(time2)

    print("Realisation totale du processus Foret - Temps de réalisation :")
    totalTime = time.time() - startime0
    print(totalTime)


def classifieurSVM(classes, test_values,x_train,y_train) : 
    startime0 = time.time()
    print("Creation du classifieur SVM - Temps de réalisation :")
    starttime = time.time()
    SVMclassifier = generateSVMClassifier(classes, x_train,y_train)
    time1= time.time()-starttime
    print(time1)

    print("Prediction du classifieur SVM- Temps de realisation : ")
    starttime = time.time()
    predictSVM(test_values, SVMclassifier)
    time2= time.time()-starttime
    print(time2)

    print("Realisation totale du processus SVM - Temps de réalisation :")
    totalTime = time.time() - startime0
    print(totalTime)


def main() :
    rcv1 = fetch_covtype()

    data = rcv1.data
    target  = rcv1.target
    classes = set(target)

    #Séparation des données entre jeu d'entraînement et de test
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
    # Generation d'un tableau contenant des tuples
    test_values = [(x_test[index],value) for index, value in enumerate(y_test)]
    
    classifieur0v0(classes, test_values,x_train,y_train)
    classifieur0vR(classes, test_values,x_train,y_train)
    foretClassifieur(classes, test_values,x_train,y_train)
    classifieurSVM(classes, test_values,x_train,y_train)
    
    

if __name__ == "__main__":
    main()