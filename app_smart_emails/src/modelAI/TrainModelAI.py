import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import os
import tensorflow as tf

class TrainModelAi:
    def __init__(self):
        self.tokenizer = Tokenizer(num_words=10000)  # maior vocabulário
        self.model = None

    def prepare_data_train(self, data_mails_train):
        subjects = [str(e["subject"]) for e in data_mails_train]
        bodies = [str(e["body"]) for e in data_mails_train]
        num_urls = np.array([e["urls"] for e in data_mails_train])
        labels = np.array([e["label"] for e in data_mails_train])

        self.tokenizer.fit_on_texts(subjects + bodies)

        X_subject = pad_sequences(self.tokenizer.texts_to_sequences(subjects), maxlen=30)  # mais comprimento
        X_body = pad_sequences(self.tokenizer.texts_to_sequences(bodies), maxlen=200)     # corpo maior

        return X_subject, X_body, num_urls, labels

    def prepare_data_test(self, mail):
        subject_seq = pad_sequences(self.tokenizer.texts_to_sequences([mail["subject"]]), maxlen=30)
        body_seq = pad_sequences(self.tokenizer.texts_to_sequences([mail["body"]]), maxlen=200)
        num_urls = np.array([[mail["urls"]]])
        return [subject_seq, body_seq, num_urls]

    def train_model(self, data_mails_train):
        X_subject, X_body, num_urls, labels = self.prepare_data_train(data_mails_train)

        X_sub_train, X_sub_val, X_body_train, X_body_val, X_urls_train, X_urls_val, y_train, y_val = train_test_split(
            X_subject, X_body, num_urls, labels, test_size=0.2, random_state=42
        )

        # --- Modelo ---
        input_subject = Input(shape=(30,))
        embed_subject = Embedding(input_dim=10000, output_dim=64)(input_subject)
        conv_subject = Conv1D(64, 3, activation="relu")(embed_subject)
        conv_subject = Dropout(0.3)(conv_subject)
        pool_subject = GlobalMaxPooling1D()(conv_subject)

        input_body = Input(shape=(200,))
        embed_body = Embedding(input_dim=10000, output_dim=64)(input_body)
        conv_body = Conv1D(64, 5, activation='relu')(embed_body)
        conv_body = Dropout(0.3)(conv_body)
        pool_body = GlobalMaxPooling1D()(conv_body)

        input_urls = Input(shape=(1,))

        concat = Concatenate()([pool_subject, pool_body, input_urls])
        dense = Dense(64, activation='relu')(concat)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.4)(dense)
        dense = Dense(32, activation='relu')(dense)
        output = Dense(1, activation='sigmoid')(dense)

        model = Model(inputs=[input_subject, input_body, input_urls], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.summary()

        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        model.fit(
            [X_sub_train, X_body_train, X_urls_train],
            y_train,
            epochs=15,         # 40k emails -> menos épocas podem ser suficientes
            batch_size=64,     # batch maior para acelerar
            validation_data=([X_sub_val, X_body_val, X_urls_val], y_val),
            callbacks=[early_stop],
            verbose=1
        )

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model.save(os.path.join(base_dir, "mailscnn.h5"))
        toke_path = os.path.join(base_dir, "tokenizer.pkl")
        with open(toke_path, "wb") as f:
            pickle.dump(self.tokenizer, f)

        self.model = model
        return

    def predict(self, data):
        if self.model is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "mailscnn.h5")
            self.model = tf.keras.models.load_model(model_path)

            toke_path = os.path.join(base_dir, "tokenizer.pkl")
            with open(toke_path, "rb") as f:
                self.tokenizer = pickle.load(f)

        data = self.prepare_data_test(data)
        pred = self.model.predict(data)
        return round(float(pred[0]),3)

