import cv2

image = cv2.imread('./images/odemgrik.png')
image = cv2.resize(image, (600, 400))
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detector = cv2.CascadeClassifier('algorithms/haarcascade_frontalface_default.xml')
results = detector.detectMultiScale(imageGray, 1.09, minNeighbors=3)

for (x, y, w, h) in results:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()