import numpy as np
import tensorflow as tf
import sklearn.model_selection as sk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense
from tensorflow.keras.utils import to_categorical

data = np.load("../data/dataset.npz", allow_pickle=True)

data_train_x = np.array(list(data["X"]), dtype=np.float32)
data_train_x = data_train_x.reshape(-1, 20, 128, 128, 1)

data_train_y = np.array(data["y"]) - 1

print(data_train_x.shape)
print(data_train_y.shape)

x_train, x_test, y_train, y_test = sk.train_test_split(
    data_train_x, data_train_y, test_size=0.2
)

# converter labels para one-hot
y_train = to_categorical(y_train, num_classes=20)
y_test = to_categorical(y_test, num_classes=20)

## Model
model = Sequential()

# CNN aplicada em cada frame
model.add(TimeDistributed(
    Conv2D(16, (3,3), activation='relu'),
    input_shape=(20, 128, 128, 1)
))

model.add(TimeDistributed(MaxPooling2D((2,2))))

model.add(TimeDistributed(
    Conv2D(32, (3,3), activation='relu'),
))

model.add(TimeDistributed(MaxPooling2D((2,2))))
model.add(TimeDistributed(Flatten()))

# LSTM aprende a sequência
model.add(LSTM(128))

# camada final
model.add(Dense(20, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

model.fit(x_train, y_train, epochs=35, batch_size=128)
model.save("model2.keras")
