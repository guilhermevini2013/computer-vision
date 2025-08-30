import os

import numpy as np
import tensorflow as tf
import cv2 as cv
import sklearn
from matplotlib import pyplot as plt

characteristics_bart_homer = {
    "mouth_homer": {
        "range": ([95, 160, 175], [140, 185, 205]),
    },
    "pants_homer": {
        "range": ([150, 95, 0], [180, 120, 90]),
    },
    "shoes_homer": {
        "range": ([24, 23, 24], [45, 45, 45]),
    },
    "shirt_bart": {
        "range": ([11, 85, 240], [50, 105, 255]),
    },
    "pants_bart": {
        "range": ([125, 0, 0], [170, 12, 20]),
    },
    "shoes_bart": {
        "range": ([125, 0, 0], [170, 12, 20]),
    }
}


def extraction_characteristic():
    path_images = "./images"
    all_images = os.listdir(path_images)

    X_pixels = []
    X_features = []
    y = []

    for image_name in all_images:
        original_image = cv.imread(f"{path_images}/{image_name}")
        h, w = original_image.shape[:2]

        # classe
        if image_name.startswith("b"):
            y.append(0)
        else:
            y.append(1)

        # ========== PIXELS ==========
        gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
        gray_image = cv.resize(gray_image, (128, 128))
        gray_image = gray_image.astype("float32") / 255.0
        gray_image = gray_image.ravel()
        X_pixels.append(gray_image)

        # ========== FEATURES ==========
        characteristic_pixel = []
        for characteristic, information in characteristics_bart_homer.items():
            min_range = np.array(information["range"][0])
            max_range = np.array(information["range"][1])

            mask = cv.inRange(original_image, min_range, max_range)
            quantity_pixels = cv.countNonZero(mask)
            percentage_pixel = round((quantity_pixels / (h * w)) * 100, 9)

            if percentage_pixel < 0.16:
                percentage_pixel = 0.0

            characteristic_pixel.append(percentage_pixel)

        X_features.append(characteristic_pixel)

    return np.array(X_pixels), np.array(X_features), np.array(y)


def create_network_and_train(x_pixels_train, x_pixels_test, x_features_train, x_features_test, y_train, y_test):

    input_pixels = tf.keras.Input(shape=(128 * 128,), name="pixels")
    dense_pixels = tf.keras.layers.Dense(512, activation="relu")(input_pixels)

    input_features = tf.keras.Input(shape=(len(characteristics_bart_homer),), name="features")
    dense_features = tf.keras.layers.Dense(32, activation="relu")(input_features)

    concatenated = tf.keras.layers.concatenate([dense_pixels, dense_features])

    x = tf.keras.layers.Dense(64, activation="relu")(concatenated)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=[input_pixels, input_features], outputs=output)

    model.summary()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(
        {"pixels": x_pixels_train, "features": x_features_train},
        y_train,
        validation_data=(
            {"pixels": x_pixels_test, "features": x_features_test},
            y_test,
        ),
        epochs=35,
        batch_size=16
    )

    model.save("bart_homer_multiinput.h5")
    return x_pixels_test, x_features_test, y_test


X_pixels, X_features, y = extraction_characteristic()

x_pixels_train, x_pixels_test, x_features_train, x_features_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X_pixels, X_features, y, test_size=0.2, random_state=1)

"""
    You must to remover this line, case you dont want train network
"""
#create_network_and_train(x_pixels_train, x_pixels_test, x_features_train, x_features_test, y_train, y_test)

model = tf.keras.models.load_model("bart_homer_multiinput.h5")

result = model.predict({"pixels": x_pixels_test, "features": x_features_test})
y_pred = (result > 0.5).astype("int32")

classes = ["bart", "homer"]

for i, img in enumerate(x_pixels_test[:10]):
    plt.imshow(img.reshape(128,128), cmap='gray')
    plt.title(f"Real: {classes[y_test[i]]} | Previsto: {classes[y_pred[i][0]]}")
    plt.axis('off')
    plt.show()
