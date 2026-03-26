import os
import cv2
import mediapipe as mp
import numpy as np

# -----------------------------
# Configurações MediaPipe
# -----------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# -----------------------------
# Caminhos e parâmetros
# -----------------------------
path = "C:\\Users\\guilherme\\Downloads\\archive"
dirs = os.listdir(path)
output_size = (224, 224)     # tamanho final do frame
num_frames_sequence = 20     # número de frames final por gesto
display_speed = 50          # ms entre frames na exibição

# -----------------------------
# Função principal
# -----------------------------
def capturar_frames_mao_completo(video_path, num_frames=num_frames_sequence, size=output_size):
    """
    Captura o movimento completo da mão (acima da cintura),
    retorna exatamente num_frames e mostra somente esses frames.
    """
    cap = cv2.VideoCapture(video_path)
    holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)

    all_frames = []
    collecting = False

    ret, frame = cap.read()
    while ret:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        # posição da cintura (média dos quadris)
        if results.pose_landmarks:
            left_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP]
            hip_y = (left_hip.y + right_hip.y) / 2
        else:
            hip_y = 0.6  # fallback se não detectar pose

        # posição da mão direita
        if results.right_hand_landmarks:
            hand_y = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].y
        else:
            hand_y = 1.0  # mão fora da tela → não coletar

        # começar/terminar coleta
        if hand_y < hip_y and not collecting:
            collecting = True
        elif hand_y >= hip_y and collecting:
            collecting = False

        if collecting:
            all_frames.append(frame)

        ret, frame = cap.read()

    cap.release()
    holistic.close()

    # -----------------------------
    # Seleção final de frames
    # -----------------------------
    total = len(all_frames)
    if total == 0:
        print(f"Nenhum frame detectado acima da cintura em {video_path}")
        return np.zeros((num_frames, size[0], size[1], 3))

    # menos frames que num_frames → repetir último frame
    if total < num_frames:
        while len(all_frames) < num_frames:
            all_frames.append(all_frames[-1])
        selected_frames = all_frames
    else:
        # mais frames → amostragem uniforme
        indices = np.linspace(0, total-1, num_frames, dtype=int)
        selected_frames = [all_frames[i] for i in indices]

    # -----------------------------
    # Redimensionar e normalizar
    # -----------------------------
    processed_frames = []
    for f in selected_frames:
        f_resized = cv2.resize(f, size)
        f_resized = f_resized / 255.0
        processed_frames.append(f_resized)

    # -----------------------------
    # Mostrar somente os frames finalizados
    # -----------------------------
    for i, f in enumerate(processed_frames):
        display_frame = (f * 255).astype(np.uint8).copy()
        cv2.putText(display_frame, f"Frame {i+1}/{num_frames}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Frames Selecionados", display_frame)
        if cv2.waitKey(display_speed) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

    return np.array(processed_frames)


# -----------------------------
# Processar todos os vídeos
# -----------------------------
for file in dirs:
    if not file.endswith(".mp4"):
        continue

    video_path = os.path.join(path, file)
    print(f"Processando: {file}")
    frames_array = capturar_frames_mao_completo(video_path)
    print(f"{file} → frames selecionados: {frames_array.shape}")