import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import itertools
import operator

#Awfull Hack to mute warning about 
# convergence issues
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def generateOvRClassifier(classes):
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

def generateOvOClassifier(classes):
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

if __name__ == "__main__":
    digits = datasets.load_digits()
    data = digits['data']
    target  = digits['target']
    classes = set(target)
    #Splitting the data to get train and test sets
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
    #Create the O v O classifiers
    o_vs_o_classifiers = generateOvOClassifier(classes)
    # We generate an array containing tuples (images,value)
    test_values = [(x_test[index],value) for index, value in enumerate(y_test)]
    # Launch the loop to predict elem in test values 
    predictOVO(test_values, o_vs_o_classifiers)
    ovrclassifier = generateOvRClassifier(classes)
    data = predictOVR(test_values,ovrclassifier)
