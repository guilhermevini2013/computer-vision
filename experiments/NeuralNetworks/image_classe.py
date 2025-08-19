import numpy as np
import tensorflow as tf
import cv2
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

images_url = os.listdir("./homer_bart")
images = []
classes = []
high = 128
width = 128

def insert_images_classes():
    for image_url in images_url:
        image = cv2.imread(f"./homer_bart/{image_url}")

        image = cv2.resize(image, (width, high))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images.append(image.ravel())
        if image_url.startswith("b"):
            classe = 0
        else:
            classe = 1
        classes.append(classe)

insert_images_classes()
x = np.asarray(images)

y = np.asarray(classes)

sk_min_max_scale = sklearn.preprocessing.MinMaxScaler()
x = sk_min_max_scale.fit_transform(x)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size=0.2, random_state=1)


def create_network_and_train():
    next_units = width*high
    network1 = tf.keras.models.Sequential()
    network1.add(tf.keras.Input(shape=(width*high,)),)

    for i in range(0,2):
        network1.add(tf.keras.layers.Dense(units=int((next_units/2)+1), activation="relu"))
        if i == 1:
            network1.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

    network1.summary()
    network1.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    network1.fit(x_train, y_train, epochs = 50)
    network1.save("bart_homer.h5")

"""
    You must to remover this line, case you dont want train network
"""
create_network_and_train()

model = tf.keras.models.load_model("bart_homer.h5")

result = model.predict(x_test)
for i,test in enumerate(y_test):
    print(test, result[i]>0.5)


