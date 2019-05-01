"""
    Ce programme permet de reconnaitre des chiffres manuscrits à partir d'un réseau de neurone
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from functools import reduce
import cv2
import numpy as np

DIGITS = datasets.load_digits()
DATA = DIGITS['data']
TARGET = DIGITS['target']

# =============================================================================
# MAIN
# =============================================================================

def main():

    # Initilisation d'une image de 10 par 64 pixels
    # Pour chaque chiffre nous préparons 64 pixels
    weight_images = [[0 for elem in range(0, 64)] for elem in range(0, 10)]
    
    # Créer des jeux de données test et train
    x_train, x_test, y_train, y_test = train_test_split(
        DATA, TARGET, test_size=0.2)
    
    for idx, elem in enumerate(x_train):
        # Calcul les nouvelles valeurs dans weight_images
        weight_images = calculate_new_weight(elem, y_train[idx], weight_images)

        #Initialise l'image que l'on va afficher
        images_display = weight_images
        #Calcul le poids maximal dans weight_images
        max_values = max(map(max, weight_images))

        images_display = [((elem/max_values)*256) for elem in images_display]

        #Arrange images_display en 5 colonne affichant chacune 2 valeurs de chiffre
        #Sur la première ligne on trouve 0,1,2,3,4 et sur le seconde 5,6,7,8,9
        vertical_1 = np.vstack((images_display[0].reshape((8, 8)), images_display[5].reshape((8, 8))))
        vertical_2 = np.vstack((images_display[1].reshape((8, 8)), images_display[6].reshape((8, 8))))
        vertical_3 = np.vstack((images_display[2].reshape((8, 8)), images_display[7].reshape((8, 8))))
        vertical_4 = np.vstack((images_display[3].reshape((8, 8)), images_display[8].reshape((8, 8))))
        vertical_5 = np.vstack((images_display[4].reshape((8, 8)), images_display[9].reshape((8, 8))))

        img = np.hstack((vertical_1, vertical_2, vertical_3,
                        vertical_4, vertical_5))

        #Affiche images_display dans une fenetre de 960 par 384 pixles
        images_display = cv2.resize(np.uint8(img), (960, 384), interpolation = cv2.INTER_AREA)

        cv2.imshow('image', np.uint8(images_display))
        
        #Attend une saisie pendant 1 millisecondes ce qui permet d'actualiser l'image en direct
        k = cv2.waitKey(1)
    
    correct_prediction = 0
    number_of_tests = 0

    #Test le modèle sur le jeu de données de test
    for idx, elem in enumerate(x_test):
        
        if predict_right_image(elem, weight_images) == y_test[idx]:
            correct_prediction += 1
        number_of_tests += 1
    #Affiche le pourcentage de prédiction pour notre modèle
    print(f"Précision du modèle : {(correct_prediction/number_of_tests) * 100}%")

    #Attend la saisie d'une touche pour terminer le programme
    k = cv2.waitKey()

# =============================================================================
# FUNCTIONS
# =============================================================================

def calculate_new_weight(image, number, weight_images):
    new_weight_images = weight_images
    
    #On vient ajouter à la partie correspondante au chiffre "number" -> image
    #Cela permet d'ajouter plus de poids aux valeurs représentant le chiffre
    new_weight_images[number] = [elem + image[idx] for idx, elem in enumerate(new_weight_images[number])]

    #Fait une prédiction à partir du modèle "weight_images" actuel
    predicted_value = predict_right_image(image, weight_images)

    #Si la prédiciton correspond à la vraie valeur, on retourne "weight_images" sans la modifier car on suppose que notre modèle est correct
    if predicted_value == number:
        return weight_images
    
    #Sinon on retourne new_weight_images pour améliorer notre modèle
    else:
        new_weight_images[predicted_value] = [elem - image[idx] for idx, elem in enumerate(new_weight_images[predicted_value])]
        return new_weight_images

#Prédit la valeur à partir de l'image d'un chiffre manuscrit et de la représentation des 10 chiffres calculés dans notre modèle
def predict_right_image(image, weight_images):
    weight_images_sum = []
    for idx, elem in enumerate(weight_images):
        weight_images_sum.append(list(map(lambda x, y: x*y, elem, image)))
        weight_images_sum[idx] = reduce(lambda x, y: x+y, weight_images_sum[idx])
    predicted_value = weight_images_sum.index(max(weight_images_sum))
    #Retourne la valeur prédite
    return predicted_value


# =============================================================================
# SCRIPT INITIATE
# =============================================================================

if __name__ == '__main__':
    main()