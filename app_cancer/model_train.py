import tensorflow as tf
import pandas as pd
import sklearn as sk

df_data = pd.read_csv("The_Cancer_data_set_V3.csv")

data = df_data.values
y_data = data[:,-1]
scaler = sk.preprocessing.StandardScaler()
data = scaler.fit_transform(data[:,:-1])

X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(data, y_data, test_size = 0.2, random_state = 66)

## Model

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(8,)))
model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(8, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(8, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(4, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=400, batch_size=4)