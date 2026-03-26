import cv2
import sklearn.model_selection as sk
from sklearn.metrics import classification_report
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
"""
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

# Carregar modelo salvo
model = load_model("model.keras")
y_pred_probs = model.predict(x_test)  # Saídas em probabilidades
y_pred = np.argmax(y_pred_probs, axis=1)  # Converter para rótulo inteiro
y_true = np.argmax(y_test, axis=1)      # Converter y_test one-hot para rótulo inteiro

# Gerar relatório de classificação
report = classification_report(y_true, y_pred, digits=4)
print("Relatório de Classificação:\n")
print(report)
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# carregar modelo
model = load_model("model2.keras")

cap = cv2.VideoCapture(0)

width, height = 128, 128
sequence = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # preprocessamento igual ao treino
    frame_resized = cv2.resize(frame, (width, height))
    frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    frame_gray = frame_gray / 255.0

    sequence.append(frame_gray)

    # manter apenas últimos 20 frames
    if len(sequence) > 20:
        sequence.pop(0)

    # quando tiver 20 frames -> prever
    if len(sequence) == 20:
        input_data = np.array(sequence, dtype=np.float32)
        input_data = input_data.reshape(1, 20, 128, 128, 1)

        prediction = model.predict(input_data, verbose=0)
        classe = np.argmax(prediction)

        cv2.putText(frame, f"Classe: {classe}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
