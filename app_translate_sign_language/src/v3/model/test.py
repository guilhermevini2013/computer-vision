import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque

# ---------- CONFIG ----------
SEQ_LEN = 25
NUM_CLASSES = 7  # ajusta se necessário

# ---------- CARREGAR MODELO ----------
model = tf.keras.models.load_model("best_foldV4_3.keras", compile=False)

# ---------- MEDIAPIPE ----------
mp_holistic = mp.solutions.holistic

# ---------- EXTRAÇÃO ----------
def extract_landmarks(results):
    def get_points(landmarks, n):
        if landmarks:
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        return np.zeros((n, 3))

    pose = get_points(results.pose_landmarks, 33)
    left = get_points(results.left_hand_landmarks, 21)
    right = get_points(results.right_hand_landmarks, 21)

    return np.concatenate([pose, left, right], axis=0)  # (75,3)

# ---------- NORMALIZAÇÃO ----------
def normalize(landmarks):
    center = landmarks[0]
    landmarks = landmarks - center

    scale = np.linalg.norm(landmarks[11] - landmarks[12])
    landmarks = landmarks / (scale + 1e-6)

    return landmarks

# ---------- BUFFER ----------
sequence = deque(maxlen=SEQ_LEN)

# ---------- LABELS (AJUSTA AQUI) ----------
labels_map = {
    0: "Acontecer",
    1: "Aluno",
    2: "Amarelo",
    3: "America",
    4: "Aproveitar",
    5: "Bala",
    6: "Banco",
}

# ---------- WEBCAM ----------
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False
) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # extrair landmarks
        landmarks = extract_landmarks(results)
        landmarks = normalize(landmarks)

        sequence.append(landmarks)

        pred_text = "Coletando..."

        # quando tiver 25 frames → prediz
        if len(sequence) == SEQ_LEN:
            seq_np = np.array(sequence)  # (25,75,3)

            # reshape igual treino
            seq_np = np.expand_dims(seq_np, axis=0)

            probs = model.predict(seq_np, verbose=0)[0]
            pred = np.argmax(probs)
            conf = np.max(probs)

            pred_text = f"{labels_map[pred]} ({conf:.2f})"

        # desenhar na tela
        cv2.putText(frame, pred_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        cv2.imshow("Sign Recognition", frame)

        # sair com Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()