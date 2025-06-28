import cv2
import dlib

image = cv2.imread('./images/odemgrik.png')
detector_facial = dlib.get_frontal_face_detector()
deteccoes = detector_facial(image, 1)
detector_pontos = dlib.shape_predictor('./algorithms/shape_predictor_68_face_landmarks.dat')

for face in deteccoes:
    points = detector_pontos(image,face)
    for point in points.parts():
        cv2.circle(image, (point.x, point.y), 2,(0,255,0), 1)
    cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), color=(0,255,0),thickness=2)
cv2.imshow("facial points", image)
cv2.waitKey(0)