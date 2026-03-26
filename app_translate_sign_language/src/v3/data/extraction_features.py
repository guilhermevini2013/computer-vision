import os

import cv2
import mediapipe as mp
import numpy as np

# =============================
# CONFIG
# =============================
NUM_KEYFRAMES = 25
MIN_GAP = 3

# =============================
# MEDIAPIPE SETUP
# =============================
mp_holistic = mp.solutions.holistic

# =============================
# FIX SEQUENCE SIZE
# =============================
def fix_sequence_length(seq, target_len=25):
    T = seq.shape[0]

    # Caso 1: mais frames → reduz
    if T > target_len:
        idx = np.linspace(0, T - 1, target_len).astype(int)
        seq = seq[idx]

    # Caso 2: menos frames → padding
    elif T < target_len:
        pad = np.zeros((target_len - T, seq.shape[1], seq.shape[2]))
        seq = np.concatenate([seq, pad], axis=0)

    return seq

# =============================
# LANDMARK EXTRACTION
# =============================
def extract_landmarks(results):
    def get_points(landmarks, n):
        if landmarks:
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        return np.zeros((n, 3))

    pose = get_points(results.pose_landmarks, 33)
    left = get_points(results.left_hand_landmarks, 21)
    right = get_points(results.right_hand_landmarks, 21)

    return np.concatenate([pose, left, right], axis=0)  # (75,3)


# =============================
# NORMALIZATION
# =============================
def normalize(landmarks):
    center = landmarks[0]
    landmarks = landmarks - center

    scale = np.linalg.norm(landmarks[11] - landmarks[12])  # ombros
    landmarks = landmarks / (scale + 1e-6)

    return landmarks


# =============================
# DELTAS
# =============================
def compute_deltas(sequence):
    return sequence[1:] - sequence[:-1]


# =============================
# MOTION SCORE
# =============================
def compute_motion_score(deltas):
    N = deltas.shape[1]

    weights = np.ones(N)
    weights[33:] = 2.0  # mãos mais importantes

    score = (np.linalg.norm(deltas, axis=2) * weights).sum(axis=1)
    return score


# =============================
# KEYFRAME SELECTION
# =============================
def select_keyframes(score, K=30, min_gap=3):
    selected = []

    for i in np.argsort(score)[::-1]:
        if all(abs(i - s) > min_gap for s in selected):
            selected.append(i)
        if len(selected) == K:
            break

    return sorted(selected)


# =============================
# AUGMENTATION
# =============================
def augment(sequence):
    def jitter(x, sigma=0.01):
        return x + np.random.normal(0, sigma, x.shape)

    def scale(x):
        s = np.random.uniform(0.9, 1.1)
        return x * s

    def rotate_z(x):
        angle = np.random.uniform(-0.1, 0.1)
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        return x @ R.T

    def dropout(x, p=0.1):
        mask = np.random.rand(*x.shape[:2]) < p
        x[mask] = 0
        return x

    sequence = jitter(sequence)
    sequence = scale(sequence)
    sequence = rotate_z(sequence)
    sequence = dropout(sequence)

    return sequence


# =============================
# MAIN PIPELINE
# =============================
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []

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

            landmarks = extract_landmarks(results)
            landmarks = normalize(landmarks)

            sequence.append(landmarks)

    cap.release()

    sequence = np.array(sequence)  # (T, N, 3)

    # --- DELTAS
    deltas = compute_deltas(sequence)

    # --- SCORE
    score = compute_motion_score(deltas)

    # --- KEYFRAMES
    idx = select_keyframes(score, NUM_KEYFRAMES, MIN_GAP)

    keyframes = sequence[idx]

    # --- GARANTIR TAMANHO FIXO
    keyframes = fix_sequence_length(keyframes, NUM_KEYFRAMES)

    # --- AUGMENT
    keyframes = augment(keyframes)

    return keyframes  # (30, 75, 3)


# =============================
# USO
# =============================

path = "C:\\Users\\guilherme\\Downloads\\archive"
dirs = os.listdir(path)

X = []
y = []

for file in dirs:
    try:
        label = int(file[:2])  # '01' → 1

        part_after_sinalizador = file.split("Sinalizador")[1]
        num_sinalizador = part_after_sinalizador.split("-")[0]
        num_gravacao = part_after_sinalizador.split("-")[1].split(".")[0]

        print("label:", label)
        print("num_sinalizador:", num_sinalizador)
        print("num_gravacao:", num_gravacao)

        data = process_video(f"{path}\\{file}")

        print("shape:", data.shape)

        # adiciona no dataset
        X.append(data)
        y.append(label)

    except Exception as e:
        print(f"Erro no arquivo {file}: {e}")

X = np.array(X)
y = np.array(y)

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)

np.savez_compressed("../model/datasetv4.npz", X=X, y=y)
print("Dataset salvo com sucesso!")