import numpy as np
from matplotlib.pyplot import imshow
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import cv2

train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                           rotation_range=10,
                                           horizontal_flip=True,
                                           zoom_range=0.2)

dataset_train = train_image_generator.flow_from_directory(directory="./images/train",
                                                          target_size=(64, 64),
                                                          batch_size=8,
                                                          class_mode="categorical",
                                                          shuffle=True)

test_image_generator = ImageDataGenerator(rescale=1. / 255)

dataset_test = test_image_generator.flow_from_directory(directory="./images/test",
                                                        target_size=(64, 64),
                                                        batch_size=1,
                                                        class_mode='categorical',
                                                        shuffle=False)

model = tf.keras.models.Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3), activation="relu", input_shape=(64,64,3)))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=32,kernel_size=(3,3), activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=32,kernel_size=(3,3), activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(units=577, activation="relu"))
model.add(Dense(units=577, activation="relu"))
model.add(Dense(units=2, activation="softmax"))
model.summary()

#model.compile("Adam", loss='categorical_crossentropy', metrics=["Accuracy"])
##model.fit(dataset_train, validation_data=dataset_test, epochs=100, batch_size=128)
##model.save("./cnn/dogs_cats.h5")

model = tf.keras.models.load_model("./cnn/dogs_cats.h5")

imagem = cv2.imread("./images/test/cat/cat.731.jpg")
cv2.imshow("animal", imagem)
cv2.waitKey(0)
imagem = cv2.resize(imagem, (64, 64))
imagem = imagem / 255
imagem = imagem.reshape(-1, 64, 64, 3)
y_predict = model.predict(imagem)
y_predict = np.argmax(y_predict)
print(dataset_test.class_indices)
print(y_predict)