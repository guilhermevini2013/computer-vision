import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler


class Network:

    def __init__(self, csv_service, scaler_X, scaler_y):
        self.csv_service = csv_service
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

    def create_train_network(self, X, y):
        num_total_params = len(X[0])
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(X.shape[1],)))
        model.add(tf.keras.layers.Dense(units=int(num_total_params / 2), activation="relu"))
        model.add(tf.keras.layers.Dense(units=int(num_total_params / 4), activation="relu"))
        model.add(tf.keras.layers.Dense(units=int(num_total_params / 6), activation="relu"))
        model.add(tf.keras.layers.Dense(units=int(num_total_params / 8), activation="relu"))
        model.add(tf.keras.layers.Dense(units=1, activation=None))

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        history =model.fit(X, y, epochs=50, batch_size=32, validation_split=0.3, callbacks=[early_stopping])
        mae_normalized = history.history['val_mae'][-1]

        mae_reais = np.expm1(self.scaler_y.inverse_transform([[mae_normalized]])[0][0])

        print(f"Erro médio absoluto em reais: R$ {mae_reais:,.2f}")
        model.save("./ann/house.h5")

    def predict_new_house(self, new_house_data):
        model = tf.keras.models.load_model("./ann/house.h5", compile=False)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        df = pd.read_csv('houses_data.csv')

        numeric_cols = ['area', 'bedroom', 'bathroom', 'garage']
        self.scaler_X.fit(df[numeric_cols])

        scaler_y = MinMaxScaler()
        y = df['price'].values.reshape(-1, 1)
        y = np.log1p(y)
        scaler_y.fit(y)

        X = self.csv_service.load_csv_and_add_new_data(new_house_data)
        result = model.predict(X)

        if result is not None:
            result_in_reais = self.revert_normalization_to_reais(result[0][0], scaler_y)

        return f"{result_in_reais:,.2f}"

    def revert_normalization_to_reais(self, normalized_value, scaler_y):
        price_real = np.expm1(scaler_y.inverse_transform([[normalized_value]])[0][0])
        return price_real