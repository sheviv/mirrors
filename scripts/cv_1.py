# Вычисление разности между кадрами

import cv2
import numpy as np

# # Вычисление разности между кадрами
# def frame_diff(prev_frame, cur_frame, next_frame):
#     # Разность между текущим и следующим кадрами
#     diff_frames_1 = cv2.absdiff(next_frame, cur_frame)
#     # Разность между текущим и предьщущим кадрами
#     diff_frames_2 = cv2.absdiff(cur_frame, prev_frame)
#     # Выполним операцию побитового И для двух разностей кадров и вернем результат
#     return cv2.bitwise_and(diff_frames_1, diff_frames_2)
# # Определение функции, получающей текущий кадр из веб-камеры
# def get_frame(cap, scaling_factor):
#     # Чтение текущего кадра из объекта захвата видео
#     _, frame = cap.read()
#     # Изменение размера изображения
#     frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
#     # Преобразование в градации серого
#     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     return gray
# # Определим основную функцию и инициализируем объект захвата видео
# if __name__ == '__main__':
#     # Определение объекта захвата видео
#     cap = cv2.VideoCapture(0)
#     # Определение масштабного множителя для изображений
#     scaling_factor = 0.5
#     # Захват текущего кадра
#     prev_frame = get_frame(cap, scaling_factor)
#     # Захват следующего кадра
#     cur_frame = get_frame(cap, scaling_factor)
#     # Захват последующего кадра
#     next_frame = get_frame(cap, scaling_factor)
#     # Чтение кадров из веб-камеры до тех пор, пока пользователь не нажмет клавишу <Esc>
#     while True:
#         # Отображение разности между кадрами
#         cv2.imshow('Object Movement', frame_diff(prev_frame, cur_frame, next_frame))
#         # Обновление переменных
#         prev_frame = cur_frame
#         cur_frame = next_frame
#         # Захват следующего кадра
#         next_frame = get_frame(cap, scaling_factor)
#         # Проверка того, не нажал ли пользователь клавишу <Esc>
#         key = cv2.waitKey(10)
#         if key == 27:
#             break
#     # Закрытие всех окон
#     cv2.destroyAllWindows()


# Отслеживание объектов с помощью цветовых пространств
# Определение функции, получающей текущий кадр из веб-камеры
# def get_frame(cap, scaling_factor):
#     # Чтение текущего кадра из объекта захвата видео
#     _, frame = cap.read()
#     # Изменение размера изображения
#     frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
#     return frame
# if __name__ == '__main__':
#     # Определение объекта захвата видео
#     cap = cv2.VideoCapture(0)
#     # Определение масштабного множителя для изображений
#     scaling_factor = 0.5
#     # Чтение кадров из веб-камеры до тех пор, пока пользователь не нажмет клавишу <Esc>
#     while True:
#         # Захват текущего кадра
#         frame = get_frame(cap, scaling_factor)
#         # Преобразуем изображение в цветовое пространство HSV
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         # Определение диапазона цветов кожи в HSV
#         lower = np.array([0, 70, 60])
#         upper = np.array([50, 150, 255])
#         # Ограничение НSV-изображения для получения только цветов кожи
#         mask = cv2.inRange(hsv, lower, upper)
#         # Вьполнение операции побитового И для маски и исходного изображения
#         img_bitwise_and = cv2.bitwise_and(frame, frame, mask=mask)
#         # Вьполнение медианного размытия
#         img_median_blurred = cv2.medianBlur(img_bitwise_and, 5)
#         # Отображение входного и выходного кадров
#         cv2.imshow('Input', frame)
#         cv2.imshow('Output', img_median_blurred)
#         # Проверка того, не нажал ли пользователь клавишу <Esc>
#         c = cv2.waitKey(5)
#         if c == 27:
#             break
#     # Закрытие всех окон
#     cv2.destroyAllWindows()


# Отслеживание объектов путем вычитания фоновых изображений
# Определение функции, захватывающей текущий кадр из веб-камеры
# def get_frame(cap, scaling_factor):
#     # Чтение текущего кадра из объекта захвата видео
#     _, frame = cap.read()
#     # Изменение размера изображения
#     frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
#     return frame
# if __name__ =='__main__':
#     # Определение объекта захвата видео
#     cap = cv2.VideoCapture(0)
#     # Определение объекта вычитания фона
#     bg_subtractor = cv2.createBackgroundSubtractorMOG2()
#     # Определим количество предьщущих кадров, которые следует
#     # использовать для обучения. Этот фактор управляет скоростью
#     # обучения алгоритма. Под скоростью обучения подразумевается
#     # скорость, с которой ваша модель будет учиться распознавать
#     # фон. Чем выше значение параметра 'history', тем ниже
#     # скорость обучения. Вы можете поэкспериментировать с этим
#     # значением, чтобы увидеть, как оно влияет на результат.
#     history = 1000
#     # Определение скорости обучения
#     learning_rate = 1.0 / history
#     # Чтение кадров из веб-камеры до тех пор,
#     # пока пользователь не нажмет клавишу <Esc>
#     while True:
#         # Захват текушего кадра
#         frame = get_frame(cap, 0.5)
#         # Вычисление маски
#         mask = bg_subtractor.apply(frame, learningRate=learning_rate)
#         # Преобразование изображения из градаций серого в пространство RGB
#         mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#         # Вывод изображений
#         cv2.imshow('Input show', frame)
#         cv2.imshow('Output show', mask & frame)
#         # Проверка го, не нажал ли пользователь клавишу <Esc>
#         c = cv2.waitKey(10)
#         if c == 27:
#             break
#     # Сброс объекта захвата видео
#     cap.release()
#     # Закрытие всех окон
#     cv2.destroyAllWindows()


# Создание интерактивного трекера объектов с помощью алгоритма CAMShift
# Определение класса, содержащего всю функциональность, необходимую для отслеживания объектов
# class ObjectTracker(object):
#     def __init__(self, scaling_factor=0.5):
#         # Инициализация объекта захвата видео
#         self.cap = cv2.VideoCapture(0)
#         # Захват кадра из веб-камеры
#         _, self.frame = self.cap.read()
#         # Масштабный множитель для захваченного изображения
#         self.scaling_factor = scaling_factor
#         # Изменение размера изображения
#         self.frame = cv2.resize(self.frame, None,
#                                 fx=self.scaling_factor, fy=self.scaling_factor,
#                                 interpolation=cv2.INTER_AREA)
#         # Создание окна для отображения кадра
#         cv2.namedWindow('Object Tracker')
#         # Установка функции обратного вызова, отслеживающей события мыши
#         cv2.setMouseCallback('Object Tracker', self.mouse_event)
#         # Инициализируем переменные для отслеживания прямоугольной рамки выбора
#         self.selection = None
#         # Инициализация переменной, связанной с начальной позицией
#         self.drag_start = None
#         # Инициализация переменной, связанной с состоянием отслеживания
#         self.tracking_state = 0
#     # Определение метода для отслеживания событий мыши
#     def mouse_event(self, event, x, y, flags, param):
#         # Преобразование координат Х и У в 16-битовые целые числа NumPy
#         x, y = np.int16([x, y])
#         # Проверка нажатия кнопки мыши
#         if event == cv2.EVENT_LBUTTONDOWN:
#             self.drag_start = (x, y)
#             self.tracking_state = 0
#         # Проверка того, не начал ли пользователь выделять область
#         if self.drag_start:
#             if flags & cv2.EVENT_FLAG_LBUTTON:
#                 # Извлечение размеров кадра
#                 h, w = self.frame.shape[:2]
#                 # Получение начальной позиции
#                 xi, yi = self.drag_start
#                 # Получение максимальной и минимальной координаты
#                 x0, y0 = np.maximum(0, np.minimum([xi, yi], [x, y]))
#                 x1, y1 = np.minimum([w, h], np.maximum([xi, yi], [x, y]))
#                 # Сброс переменной selection
#                 self.selection = None
#                 # Завершение выделения прямоугольной области
#                 if x1 - x0 > 0 and y1 - y0 > 0:
#                     self.selection = (x0, y0, x1, y1)
#             else:
#                 # Если выделение завершено, начать отслеживание
#                 self.drag_start = None
#                 if self.selection is not None:
#                     self.tracking_state = 1
#     # Метод, начинающий отслеживание объекта
#     def start_tracking(self):
#         # Итерируем до тех пор, пока пользователь не нажмет клавишу <Esc>
#         while True:
#             # Захват кадра из веб-камеры
#             _, self.frame = self.cap.read()
#             # Изменение размера входного кадра
#             self.frame = cv2.resize(self.frame, None,
#                                     fx=self.scaling_factor, fy=self.scaling_factor,
#                                     interpolation=cv2.INTER_AREA)
#             # Создание копии кадра
#             vis = self.frame.copy()
#             # Преобразуем цветовое пространство кадра из RGB в HSV
#             hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
#             # Создание маски на основании предварительно установленных пороговых значений
#             mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
#             # Проверка выделения пользователем области
#             if self.selection:
#                 # Извлечение координат выделенного прямоугольника
#                 x0, y0, x1, y1 = self.selection
#                 # Извлечем окно отслеживания
#                 self.track_window = (x0, y0, x1 - x0, y1 - y0)
#                 # Извлечение интересующей нас области
#                 hsv_roi = hsv[y0:y1, x0:x1]
#                 mask_roi = mask[y0:y1, x0:x1]
#                 # Compute the histogram of the region of interest in the HSV image using the mask
#                 hist = cv2.calcHist([hsv_roi], [0], mask_roi, [16], [0, 180])
#                 # Normalize and reshape the histogram
#                 cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
#                 self.hist = hist.reshape(-1)
#                 # Extract the region of interest from the frame
#                 vis_roi = vis[y0:y1, x0:x1]
#                 # Compute the image negative (for display only)
#                 cv2.bitwise_not(vis_roi, vis_roi)
#                 vis[mask == 0] = 0
#             # Check if the system in the "tracking" mode
#             if self.tracking_state == 1:
#                 # Reset the selection variable
#                 self.selection = None
#                 # Compute the histogram back projection
#                 hsv_backproj = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
#                 # Compute bitwise AND between histogram backprojection and the mask
#                 hsv_backproj &= mask
#                 # Define termination criteria for the tracker
#                 term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
#                 # Apply CAMShift on 'hsv_backproj'
#                 track_box, self.track_window = cv2.CamShift(hsv_backproj, self.track_window, term_crit)
#                 # Draw an ellipse around the object
#                 cv2.ellipse(vis, track_box, (0, 255, 0), 2)
#             # Show the output live video
#             cv2.imshow('Object Tracker', vis)
#             # Stop if the user hits the 'Esc' key
#             c = cv2.waitKey(5)
#             if c == 27:
#                 break
#         # Close all the windows
#         cv2.destroyAllWindows()
# if __name__ == '__main__':
#     # Start the tracker
#     ObjectTracker().start_tracking()



# Define a function to track the object
# def start_tracking():
#     # Initialize the video capture object
#     cap = cv2.VideoCapture(0)
#
#     # Define the scaling factor for the frames
#     scaling_factor = 0.5
#     scaling_factor = 1
#
#     # Number of frames to track
#     num_frames_to_track = 5
#
#     # Skipping factor
#     num_frames_jump = 2
#
#     # Initialize variables
#     tracking_paths = []
#     frame_index = 0
#
#     # Define tracking parameters
#     tracking_params = dict(winSize=(11, 11), maxLevel=2,
#                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
#                                      10, 0.03))
#     # Iterate until the user hits the 'Esc' key
#     while True:
#         # Capture the current frame
#         _, frame = cap.read()
#
#         # Resize the frame
#         frame = cv2.resize(frame, None, fx=scaling_factor,
#                            fy=scaling_factor, interpolation=cv2.INTER_AREA)
#
#         # Convert to grayscale
#         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#         # Create a copy of the frame
#         output_img = frame.copy()
#
#         if len(tracking_paths) > 0:
#             # Get images
#             prev_img, current_img = prev_gray, frame_gray
#
#             # Organize the feature points
#             feature_points_0 = np.float32([tp[-1] for tp in \
#                                            tracking_paths]).reshape(-1, 1, 2)
#
#             # Compute optical flow
#             feature_points_1, _, _ = cv2.calcOpticalFlowPyrLK(
#                 prev_img, current_img, feature_points_0,
#                 None, **tracking_params)
#
#             # Compute reverse optical flow
#             feature_points_0_rev, _, _ = cv2.calcOpticalFlowPyrLK(
#                 current_img, prev_img, feature_points_1,
#                 None, **tracking_params)
#
#             # Compute the difference between forward and
#             # reverse optical flow
#             diff_feature_points = abs(feature_points_0 - \
#                                       feature_points_0_rev).reshape(-1, 2).max(-1)
#
#             # Extract the good points
#             good_points = diff_feature_points < 1
#
#             # Initialize variable
#             new_tracking_paths = []
#
#             # Iterate through all the good feature points
#             for tp, (x, y), good_points_flag in zip(tracking_paths,
#                                                     feature_points_1.reshape(-1, 2), good_points):
#                 # If the flag is not true, then continue
#                 if not good_points_flag:
#                     continue
#
#                 # Append the X and Y coordinates and check if
#                 # its length greater than the threshold
#                 tp.append((x, y))
#                 if len(tp) > num_frames_to_track:
#                     del tp[0]
#
#                 new_tracking_paths.append(tp)
#
#                 # Draw a circle around the feature points
#                 cv2.circle(output_img, (x, y), 3, (0, 255, 0), -1)
#
#             # Update the tracking paths
#             tracking_paths = new_tracking_paths
#
#             # Draw lines
#             cv2.polylines(output_img, [np.int32(tp) for tp in \
#                                        tracking_paths], False, (0, 150, 0))
#
#         # Go into this 'if' condition after skipping the
#         # right number of frames
#         if not frame_index % num_frames_jump:
#             # Create a mask and draw the circles
#             mask = np.zeros_like(frame_gray)
#             mask[:] = 255
#             for x, y in [np.int32(tp[-1]) for tp in tracking_paths]:
#                 cv2.circle(mask, (x, y), 6, 0, -1)
#
#             # Compute good features to track
#             feature_points = cv2.goodFeaturesToTrack(frame_gray,
#                                                      mask=mask, maxCorners=500, qualityLevel=0.3,
#                                                      minDistance=7, blockSize=7)
#
#             # Check if feature points exist. If so, append them
#             # to the tracking paths
#             if feature_points is not None:
#                 for x, y in np.float32(feature_points).reshape(-1, 2):
#                     tracking_paths.append([(x, y)])
#
#         # Update variables
#         frame_index += 1
#         prev_gray = frame_gray
#
#         # Display output
#         cv2.imshow('Optical Flow', output_img)
#
#         # Check if the user hit the 'Esc' key
#         c = cv2.waitKey(1)
#         if c == 27:
#             break
#
#
# if __name__ == '__main__':
#     # Start the tracker
#     start_tracking()
#
#     # Close all the windows
#     cv2.destroyAllWindows()



# Load the Haar cascade file
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Check if the cascade file has been loaded correctly
# if face_cascade.empty():
# 	raise IOError('Unable to load the face cascade classifier xml file')

# Initialize the video capture object
cap = cv2.VideoCapture(0)
# Define the scaling factor
# Iterate until the user hits the 'Esc' keyq
while True:
    # Capture the current frame
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        # print(f"f: {1111_4444_4444_2222}")
        break
# Release the video capture object
cap.release()
