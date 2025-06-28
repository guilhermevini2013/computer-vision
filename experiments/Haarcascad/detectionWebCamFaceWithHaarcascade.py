import cv2

webCamImage = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('algorithms/haarcascade_frontalface_default.xml')

while True:
    ret, frame = webCamImage.read()
    frame = cv2.resize(frame, (600, 400))
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(frameGray, scaleFactor=1.099, minNeighbors=3, minSize=(70, 70), maxSize=(170, 170))

    for (x, y, w, h) in faces:
        print(w, h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
webCamImage.release()
