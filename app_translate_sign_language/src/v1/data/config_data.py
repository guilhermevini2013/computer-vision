import os
import cv2
import numpy as np

path = "C:\\Users\\guilherme\\Downloads\\archive"
dirs = os.listdir(path)

X = []
y = []

def capturar(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width, height = 128, 128
    features = []

    indices = np.linspace(0, total_frames - 1, 20, dtype=int)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame / 255.0

        features.append(frame)

    cap.release()

    return np.array(features)

for file in dirs:
    try:
        label = int(file[:2])
    except:
        continue

    video_path = f"{path}\\{file}"

    print("Processando:", file)

    features = capturar(video_path)

    if len(features) == 20:
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

np.savez_compressed("dataset.npz", X=X, y=y)

print("Dataset salvo!")
print("Shape X:", X.shape)
print("Shape y:", y.shape)
""" 
# Inicializa MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Cria os objetos de rastreamento
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, smooth_segmentation=True)

# Captura da webcam

path = "C:\\Users\\guilherme\\Downloads\\archive"
dirs = os.listdir(path)

def capturar(path):
    final_results_hand = []
    final_results_pose = []

    cap = cv2.VideoCapture(path)

    width, height = 64, 64

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensiona o frame da câmera (opcional)
        frame = cv2.resize(frame, (width, height))

        # Converte para RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Processa mãos, pose
        results_hands = hands.process(image)
        results_pose = pose.process(image)


        if results_pose.pose_landmarks:
            for content in results_pose.pose_landmarks.landmark:
                if content is None:
                    continue
                final_results_pose.append([content.x, content.y])

        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    final_results_hand.append([lm.x, lm.y])

        # Desenha Pose
        
        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results_pose.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # Desenha Mãos
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
        

        # Mostra o resultado final
        cv2.imshow("Hands + Pose + Face (preto)", frame)

        # Sai com ESC
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return final_results_hand, final_results_pose

for file in dirs:
    y = file[:2]  # '01'

    part_after_sinalizador = file.split("Sinalizador")[1]  # '01-1.mp4'
    num_sinalizador = part_after_sinalizador.split("-")[0]  # '01'

    num_gravacao = part_after_sinalizador.split("-")[1].split(".")[0]  # '1'

    print("y:", y)
    print("num_sinalizador:", num_sinalizador)
    print("num_gravacao:", num_gravacao)

    if num_sinalizador == "12" and num_gravacao == "3":
        final_results_hand, final_results_pose = capturar(f"{path}\\{file}")

        # Cria um DataFrame temporário com a nova linha
        new_row = pd.DataFrame([{
            "X_hands": final_results_hand,
            "X_pose": final_results_pose,
            "Y": y
        }])

        # Concatena com o df_data_train
        df_data_train = pd.concat([df_data_train, new_row], ignore_index=True)


df_data_train.to_csv("train.csv", index=False)
df_data_test.to_csv("test.csv", index=False)
        
"""