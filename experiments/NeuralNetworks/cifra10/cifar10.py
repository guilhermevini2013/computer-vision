import numpy as np
import tensorflow as tf
import sklearn
from tensorflow.keras.datasets import cifar10
import cv2
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = np.asarray([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_train])
x_test = np.asarray([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in x_test])

def normalize_data(array_image):
    nsamples, nx, ny = array_image.shape
    array_image = array_image.reshape((nsamples, nx * ny))

    sk_min_max_scale = sklearn.preprocessing.MinMaxScaler()
    return sk_min_max_scale.fit_transform(array_image)

x_train_normal = normalize_data(x_train)
units_number = int((x_train_normal.shape[1]/2) + 1)

def create_train_network():
    network = tf.keras.models.Sequential()
    network.add(tf.keras.Input(shape=(x_train_normal.shape[1],)))

    network.add(tf.keras.layers.Dense(units=units_number, activation="relu"))
    network.add(tf.keras.layers.Dense(units=int((units_number/2)+1), activation="relu"))
    network.add(tf.keras.layers.Dense(units=int((units_number / 2) + 1), activation="relu"))
    network.add(tf.keras.layers.Dense(units=int((units_number / 2) + 1), activation="relu"))
    network.add(tf.keras.layers.Dense(units=int((units_number / 2) + 1), activation="relu"))
    network.add(tf.keras.layers.Dense(units=int((units_number / 2) + 1), activation="relu"))
    network.add(tf.keras.layers.Dense(units=10, activation="softmax"))

    network.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    network.fit(epochs = 130, x = x_train_normal, y = y_train,batch_size=520)
    network.save("cifar10.h5")

"""
    You must to remover this line, case you don`t want train network
"""
#create_train_network()

model_cifra = tf.keras.models.load_model("cifar10.h5")

x_test_normal = normalize_data(x_test)
y_test_flat = y_test.flatten()

pred = model_cifra.predict(x_test_normal)
y_pred = np.argmax(pred, axis=1)

classes = [
    'avião', 'automóvel', 'pássaro', 'gato', 'veado',
    'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão'
]

for i, img in enumerate(x_test[:10]):
    plt.imshow(img, cmap='gray')
    plt.title(f"Real: {classes[y_test[i][0]]} | Previsto: {classes[y_pred[i]]}")
    plt.axis('off')
    plt.show()