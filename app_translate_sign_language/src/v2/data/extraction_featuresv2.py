import os
import cv2
import mediapipe as mp
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------
# Configuração MediaPipe
# ----------------------------
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

path = "C:\\Users\\guilherme\\Downloads\\archive"
dirs = os.listdir(path)

# ----------------------------
# Função para processar cada vídeo
# ----------------------------
def capturar(video_path):
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.3
    )

    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.3,
        smooth_segmentation=True
    )

    final_results_hand = []
    final_results_pose = []

    cap = cv2.VideoCapture(video_path)
    width, height = 820, 580

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results_hands = hands.process(image)
        results_pose = pose.process(image)

        frame_data_hand = []
        frame_data_pose = []

        if results_pose.pose_landmarks:
            for lm in results_pose.pose_landmarks.landmark:
                frame_data_pose.append((lm.x, lm.y))
            left_hip = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            cx = (left_hip.x + right_hip.x) / 2
            cy = (left_hip.y + right_hip.y) / 2
        else:
            cx = 0.5
            cy = 0.6

        default_hand = [(cx, cy)] * 21

        detected_hands = []
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                detected_hands.append([(lm.x, lm.y) for lm in hand_landmarks.landmark])

        if len(detected_hands) == 2:
            frame_data_hand = detected_hands[0] + detected_hands[1]
        elif len(detected_hands) == 1:
            frame_data_hand = detected_hands[0] + default_hand
        else:
            frame_data_hand = default_hand + default_hand

        final_results_hand.append(frame_data_hand)
        final_results_pose.append(frame_data_pose)

    cap.release()

    def reduzir_para_20(lista):
        total = len(lista)
        if total >= 20:
            indices = np.linspace(0, total - 1, 20, dtype=int)
            return [lista[i] for i in indices]
        else:
            resultado = lista.copy()
            while len(resultado) < 20:
                resultado.append(lista[-1] if lista else [(0, 0)])
            return resultado

    final_results_hand = reduzir_para_20(final_results_hand)
    final_results_pose = reduzir_para_20(final_results_pose)

    seq_hand = [np.array(f).flatten() for f in final_results_hand]
    seq_pose = [np.array(f).flatten() for f in final_results_pose]

    return seq_hand, seq_pose

# ----------------------------
# Função para processar arquivo
# ----------------------------
def processar_arquivo(file):
    y = int(file[:2]) - 1
    seq_hand, seq_pose = capturar(os.path.join(path, file))
    return seq_hand, seq_pose, y

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    # Necessário no Windows para multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()

    X_hand = []
    X_pose = []
    y_dataset = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(processar_arquivo, file): file for file in dirs}

        for future in as_completed(futures):
            seq_hand, seq_pose, y = future.result()
            X_hand.append(seq_hand)
            X_pose.append(seq_pose)
            y_dataset.append(y)

    X_hand = np.array(X_hand)
    X_pose = np.array(X_pose)
    y_dataset = np.array(y_dataset)

    print("Shape X_hand:", X_hand.shape)
    print("Shape X_pose:", X_pose.shape)
    print("Shape y:", y_dataset.shape)

    np.savez_compressed("datasetv2.npz", X_hand=X_hand, X_pose=X_pose, y=y_dataset)
    print("Dataset salvo com sucesso 🚀")