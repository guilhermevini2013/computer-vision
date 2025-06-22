import os
import cv2
import numpy as np
from PIL import Image


class RecognitionFacialService:
    def __init__(self):
        super().__init__()
        self.lbph_recognizer = cv2.face.LBPHFaceRecognizer.create()

    def train_recognitive(self):
        def get_ids_faces():
            dirs = os.listdir('faces_to_train')
            if len(dirs) == 0:
                raise Exception("No face found in the database.")
            ids = []
            faces = []
            id = 1
            for dir in dirs:
                images = os.listdir(f'faces_to_train/{dir}')
                for image in images:
                    image_gray = Image.open(f'faces_to_train/{dir}/{image}')
                    image_np = np.array(image_gray, dtype='uint8')
                    ids.append(id)
                    faces.append(image_np)
                id += 1
            return np.array(ids), faces

        ids, faces = get_ids_faces()
        lbph_recognizer_path = 'algorithms/lbphRecognizer.xml'
        self.lbph_recognizer.train(faces, ids)
        self.lbph_recognizer.write(lbph_recognizer_path)

    def get_face_recognitive(self, frame, total_frame, xy_face):
        if os.path.exists("./algorithms/lbphRecognizer.xml"):
            self.lbph_recognizer.read("./algorithms/lbphRecognizer.xml")
            frame_np = np.array(frame, dtype='uint8')
            id, predict = self.lbph_recognizer.predict(frame_np)
            x, y = xy_face
            prediction_zero_ten = round((100 - predict) / 10, 2)
            cv2.putText(
                total_frame,
                text=f'ID: {id}',
                org=(x - 10, y - 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(0, 255, 0),
                thickness=2
            )

            cv2.putText(
                total_frame,
                text=f'Prediction: {prediction_zero_ten}',
                org=(x - 10, y - 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(0, 255, 0),
                thickness=2
            )
            return total_frame
        else:
            raise Exception("No algorithm found, please train the model.")