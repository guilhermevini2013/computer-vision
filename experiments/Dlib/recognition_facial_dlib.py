import os

import dlib
import numpy as np
from PIL import Image

facial_detector = dlib.get_frontal_face_detector()
points_detector = dlib.shape_predictor('./algorithms/shape_predictor_68_face_landmarks.dat')
facial_descriptor = dlib.face_recognition_model_v1('./algorithms/dlib_face_recognition_resnet_model_v1.dat')

all_descriptors = None

images_path = [os.path.join('./images/recognition',f) for f in os.listdir('./images/recognition')]

for path in images_path:
    print(path)
    image = Image.open(path).convert('RGB')
    image_np = np.array(image, 'uint8')
    face_found = facial_detector(image_np, 1)
    try:
        if len(face_found) < 1:
            raise Exception(f"face found not at: {path}")
        if len(face_found) > 1:
            raise Exception(f"A lot of faces found at: {path}")

        face = face_found[0]
        points = points_detector(image_np, face)
        descriptors = facial_descriptor.compute_face_descriptor(image_np, points)
        descriptor = np.asarray([d for d in descriptors], dtype=np.float64)
        descriptor = descriptor[np.newaxis, :]

        if all_descriptors is None:
            all_descriptors = descriptor
        else:
            all_descriptors = np.concatenate((all_descriptors, descriptor), axis=0)
    except Exception as e:
        print(e)
distance = np.linalg.norm(all_descriptors[1] - all_descriptors[2])
print(distance)