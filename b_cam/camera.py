import cv2

capture = cv2.VideoCapture("sample_movies/dark_tunnel.mp4")

while(True):
    ret,frame = capture.read()
    w_size = (300,200)
    frame = cv2.resize(frame,w_size)

    cv2.imshow("title",frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
