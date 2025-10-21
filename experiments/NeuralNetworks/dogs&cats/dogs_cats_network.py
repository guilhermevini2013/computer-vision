import os
import cv2 as cv
import numpy as np
import sklearn as sk
import tensorflow as tf

dirs = os.listdir("./dataset")


def recover_classes_pixels():
    pixels = []
    classes = []

    for path_image in dirs:
        image_original = cv.imread(f"./dataset/{path_image}")
        image_gray = cv.cvtColor(image_original, cv.COLOR_BGRA2GRAY)
        image_gray = cv.resize(image_gray, (64, 64))

        ## normalize 0 until 1
        image_gray = image_gray.astype("float32") / 255.0

        pixels.append(image_gray.ravel())

        if path_image.startswith("c"):
            classes.append(0)
        else:
            classes.append(1)

    return np.asarray(pixels), np.asarray(classes)


def create_train_network(pixels_train, pixels_test, classes_train, classes_test):
    network1 = tf.keras.models.Sequential()

    network1.add(tf.keras.layers.Input(shape=(64 * 64,)))

    network1.add(tf.keras.layers.Dense(units=2048, activation="relu"))
    network1.add(tf.keras.layers.Dropout(0.4))

    network1.add(tf.keras.layers.Dense(units=1024, activation="relu"))
    network1.add(tf.keras.layers.Dropout(0.3))

    network1.add(tf.keras.layers.Dense(units=512, activation="relu"))
    network1.add(tf.keras.layers.Dropout(0.2))

    network1.add(tf.keras.layers.Dense(units=256, activation="relu"))
    network1.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

    print(network1.summary())

    network1.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    network1.fit(
        x=pixels_train,
        y=classes_train,
        epochs=120,
        batch_size=64,
        validation_data=(pixels_test, classes_test)
    )
    return


pixels, classes = recover_classes_pixels()

pixels_train, pixels_test, classes_train, classes_test = sk.model_selection.train_test_split(pixels, classes,
                                                                                             test_size=0.2,
                                                                                             random_state=1)
create_train_network(pixels_train, pixels_test, classes_train, classes_test)

print(pixels[0], classes[0])
