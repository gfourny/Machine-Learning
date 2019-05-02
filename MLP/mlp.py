'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 20

# Sépare les data en deux jeux de données: test et train
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Transforme les donnés x qui sont en trois dimensions (60000, 28 ,28) en deux dimension de taille(60000, 784)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
# Converti toutes les données en float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# On divise par 255 (correspondant à la couleur) le poids des données dans les jeux de données x
x_train /= 255
x_test /= 255
# Affiche le nombre de valeur dans le jeu de train et le jeu de test
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Converti les valeurs des données dans y de integer vers binaire
# Une liste contien 10 objets, les valeurs sont toutes à 0 sauf pour la position correspondant au chiffre qui sera à 1
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Défini le model comme séquenciel
model = Sequential()

# Le model va mainteannt prendre en entré une liste de shape (*, 784)
# et en sortie une liste de shape (*, 512)
model.add(Dense(512, activation='relu', input_shape=(784,)))
# Le model va mainteannt retourner une liste de shape (*, 10) correspondant à la variable num_classes (le nombre de chiffre que nous avons de 0 à 9)
model.add(Dense(num_classes, activation='softmax'))
# Affiche une représentation du modèle
model.summary()

# Compilation de notre modèle
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# Entrainement de notre modèle sur les jeux de données train et valdiation à partir des jeux de données test
# Epochs correspond au nombre de passage que l'on souhaite faire
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

# Calcul le pourcentage de précision de notre modèle
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {score[0]*100} %")
print(f"Test accuracy: {score[1]*100} %")