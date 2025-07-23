import cv2

tracking = cv2.TrackerKCF.create()

video_futebol = cv2.VideoCapture("./video/futebol (online-video-cutter.com).mp4")
ok, frame = video_futebol.read()

bbox = cv2.selectROI("Select Object", frame)
cv2.destroyWindow("Select Object")

tracking.init(frame, bbox)

fps = video_futebol.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

while True:
    ok, frame = video_futebol.read()

    if not ok:
        break
    success, bbox = tracking.update(frame)

    if success:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking failed", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.imshow("mario", frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

video_futebol.release()
cv2.destroyAllWindows()
