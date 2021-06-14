import cv2
import time

# window = "OpenCV_window"
cam = cv2.VideoCapture(2)
# print(cam, "is open:", cam.isOpened())

# cv2.namedWindow(window)
# ready, frame = cam.read()
# print("Frame ready:", ready)

# cv2.imshow(window, frame)
# cv2.waitKey(1)
# input("Press ENTER to close window and camera.")

# cv2.destroyAllWindows()
# cam.release()

# light on Logitech C922 webcam remains on at this point

input("Press ENTER to exit Python.")

# light on webcam goes off only after Python exits

while (cam.isOpened()):
    while True:
        # cv2.namedWindow(window)
        # ready, frame = cam.read()
        # print("Frame ready:", ready)

        # cv2.imshow(window, frame)
        # cv2.waitKey(1)
        # input("Press ENTER to close window and camera.")

        cv2.destroyAllWindows()
        cam.release()

        ret, img = cam.read()
        cv2.imshow('img', img)
        if cv2.waitKey(30) & 0xff == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
else:
    print("Alert ! Camera disconnected")
