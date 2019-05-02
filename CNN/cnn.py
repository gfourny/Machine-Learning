'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# Défini les dimensions des images
img_rows, img_cols = 28, 28

# Sépare les data en deux jeux de données: test et train
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print (K.image_data_format())
# Redimensione x_train et x_test en 4 dimensions
# 1. on conserve la 1er dimension 
# 2. on ajoute une dimension égale à 1
# 3 et 4 on utilise les dimensions de l'image definies plus tôt
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Converti toutes les données en float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# On divise par 255 (correspondant à la couleur) le poids des données dans les jeux de données x
x_train /= 255
x_test /= 255
# Affiche le nombre de valeur dans le jeu de train et le jeu de test
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Converti les valeurs des données dans y de integer vers binaire
# Une liste contien 10 objets, les valeurs sont toutes à 0 sauf pour la position correspondant au chiffre qui sera à 1
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Défini le model comme séquenciel
model = Sequential()


model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
# Le model va mainteannt retourner une liste de shape (*, 128)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# Le model va mainteannt retourner une liste de shape (*, 10) correspondant à la variable num_classes (le nombre de chiffre que nous avons de 0 à 9)
model.add(Dense(num_classes, activation='softmax'))

# Compilation de notre modèle
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Entrainement de notre modèle sur les jeux de données train et valdiation à partir des jeux de données test
# Epochs correspond au nombre de passage que l'on souhaite faire
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
# Calcul le pourcentage de précision de notre modèle
print(f"Test loss: {score[0]*100} %")
print(f"Test accuracy: {score[1]*100} %")