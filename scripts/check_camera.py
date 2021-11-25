# Найти id камер, подключенных к ПК
# Запуск камеры

import cv2
import numpy as np

all_camera_idx_available = []
for camera_idx in range(10):
    cap = cv2.VideoCapture(camera_idx)
    if cap.isOpened():
        print(f'Camera index available: {camera_idx}')
        all_camera_idx_available.append(camera_idx)
        cap.release()

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    while True:
        ret, img = cap.read()
        cv2.imshow('img', img)
        if cv2.waitKey(30) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Alert ! Camera disconnected")
