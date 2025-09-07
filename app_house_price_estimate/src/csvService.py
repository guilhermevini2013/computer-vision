import pandas as pd
import numpy as np


class CsvService:
    def __init__(self, scaler_X, scaler_y):
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

    def normalize_data(self, csv_path):
        df_encoded = pd.read_csv(csv_path)
        numeric_cols = ['area', 'bedroom', 'bathroom', 'garage']

        X = df_encoded.drop('price', axis=1)
        y = df_encoded['price'].values.reshape(-1, 1)
        y = np.log1p(y)
        X[numeric_cols] = self.scaler_X.fit_transform(X[numeric_cols])
        self.scaler_y.fit(y)
        y = self.scaler_y.transform(y)
        X = np.asarray(X)

        return X, y

    def load_csv_and_add_new_data(self, new_house_data):
        df = pd.read_csv('houses_data.csv')

        df_new = pd.DataFrame([new_house_data])
        df_new['neighborhood'] = df_new[
            'neighborhood'].str.strip().str.lower()
        df_new_encoded = pd.get_dummies(df_new, columns=['neighborhood'])

        neighborhood_columns = [col for col in df.columns if 'neighborhood_' in col]

        for col in neighborhood_columns:
            if col in df_new_encoded.columns:
                df_new_encoded[col] = 1
            else:
                df_new_encoded[col] = 0

        missing_cols = set(df.columns) - set(df_new_encoded.columns)
        for col in missing_cols:
            df_new_encoded[col] = 0

        df_new_encoded = df_new_encoded[df.columns]

        df_new_encoded = df_new_encoded.drop(columns=['price'], errors='ignore')

        numeric_cols = ['area', 'bedroom', 'bathroom', 'garage']
        df_new_encoded[numeric_cols] = self.scaler_X.transform(df_new_encoded[numeric_cols])

        return np.asarray(df_new_encoded)