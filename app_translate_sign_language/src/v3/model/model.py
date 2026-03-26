import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
from sklearn.model_selection import StratifiedKFold

# ---------- Config ----------
NUM_CLASSES = 7
SEQ_LEN = 25
BATCH = 16
EPOCHS = 60
SEED = 46

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ---------- 1. Carregar dataset ----------
data = np.load("datasetv5_noaugment.npz")

X = data["X"]  # esperado: (N, 25, 75, 3)
y = data["y"]  # pode ser (N,) ou (N,7)
y = y - 1
print("Shape X:", X.shape)
print("Shape y:", y.shape)

# ---------- 2. Corrigir labels ----------
if len(y.shape) == 1:
    y_labels = y
    y = tf.keras.utils.to_categorical(y, NUM_CLASSES)
else:
    y_labels = np.argmax(y, axis=1)

# ---------- 3. Normalização ----------
# (opcional mas ajuda estabilidade)
X = X.astype(np.float32)


# ---------- 4. tf.data ----------
def make_tf_dataset(X, y, training=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))

    if training:
        ds = ds.shuffle(2000)

    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


# ---------- 5. Modelo (OTIMIZADO PRA KEYPOINTS) ----------
def build_model():
    inp = layers.Input(shape=(SEQ_LEN, 75, 3))

    # achata keypoints
    x = layers.Reshape((SEQ_LEN, 225))(inp)

    # normalização temporal
    x = layers.LayerNormalization()(x)

    # GRU stack
    x = layers.GRU(128, return_sequences=True, dropout=0.3)(x)
    x = layers.GRU(64, dropout=0.3)(x)

    # head
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = models.Model(inp, out)
    model.summary()
    return model


# ---------- 6. Treino K-Fold ----------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_labels)):
    print(f"\n=== Fold {fold + 1} ===")

    X_train, X_val = X[tr_idx], X[val_idx]
    y_train, y_val = y[tr_idx], y[val_idx]

    train_ds = make_tf_dataset(X_train, y_train, True)
    val_ds = make_tf_dataset(X_val, y_val, False)

    model = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    cbs = [
        callbacks.EarlyStopping(patience=6, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=3, factor=0.4, min_lr=1e-5),
        callbacks.ModelCheckpoint(
            filepath=f'best_foldV4_{fold}_noaugment.keras',  # formato nativo Keras
            save_best_only=True,
            verbose=1
        )
    ]

    model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=cbs
    )


# ---------- 7. Inferência ----------
def predict_sample(sample):
    """
    sample: (25, 75, 3)
    """
    sample = np.expand_dims(sample, 0)
    prob = model.predict(sample)[0]
    return np.argmax(prob), float(np.max(prob))

# exemplo:
# pred, conf = predict_sample(X[0])
# print("Classe:", pred, "Confiança:", conf)