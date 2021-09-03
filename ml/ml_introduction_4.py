# https://timeseries-ru.github.io/course/day_4.html
# Курс “Введение в анализ данных и машинное обучение”

# нейронные сети с помощью библиотеки keras, устройство нейросетей,
# сети, которые выдают свой вход и зачем они нужны - автоэнкодеры,
# элементы работы с изображениями: классификация, сегментация
# предобученные сети работы с изображениями,
# тексты: кодирование, поиск “по смыслу”, извлечение сущностей, суммаризация,
# многомерные временные ряды (и рекуррентные нейронные сети)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


"""
1 Нейронные сети на keras
"""
# различные слои подходят под различные задачи
import os

# будем учить сети на CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 0 для GPU
import warnings
import tensorflow.keras as keras
warnings.filterwarnings('ignore')

# датасет - Forest Cover Types - покрытие лесов/ 54 признака лесного покрытия, и метку - один из 7 типов леса
from sklearn.utils import shuffle
from sklearn.datasets import fetch_covtype
# X, y = fetch_covtype(return_X_y=True)
# train = int(len(y) * 0.8)
# indices = shuffle(list(range(len(y))), random_state=1)
# train_indices = indices[:train]
# test_indices = indices[train:]
# print("Размер всех данных %d, тренировочных %d" % (len(y), train))

# классов более двух - сеть должна отдавать вектор(неоткалиброванных вероятностей),
# где индекс наибольшего числа указывает на предсказанный класс(softmax)
# def create_model(number_features, number_classes):
#     model = keras.Sequential([
#         keras.layers.Dense(units=256, activation='relu', input_shape=(number_features,)),
#         # пятую часть нейронов при тренировке будем занулять
#         keras.layers.Dropout(0.2),
#         # промежуточный слой
#         keras.layers.Dense(32, activation='relu'),
#         # выходной слой
#         keras.layers.Dense(number_classes, activation='softmax')])
#     # мы должны специфицировать задачу сети
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler().fit(X[train_indices])
# X_train = scaler.transform(X[train_indices])
# X_test = scaler.transform(X[test_indices])
# мы должны привести наши классы к векторам вида (0, 1, 0) где 1 стоит на том месте, где должен быть нужный пол
# y_categorical = keras.utils.to_categorical(y)
# y_train = y_categorical[train_indices]
# y_test = y_categorical[test_indices]
# перед созданием модели сбросим уже сохраненные модели
# keras.backend.clear_session()
# model = create_model(number_features=X_train.shape[1], number_classes=y_train.shape[1])
# в compile:
# 1. optimizer - способ поиска минимума функции ошибки принимают решение, как далеко шагать с помощью вычисленной производной ошибки,самые используемые:
# - rmsprop(обычно для рекуррентных сетей),
# - sgd (когда данных очень много), adam (один из самых лучших)
# 2. loss - функция ошибки. Для бинарной классификации используют на последнем слое активацию sigmoid и loss = 'binary_crossentropy',
# при многоклассовой классификации - 'categorical_crossentropy'. Кросс-энтропия тем ниже, чем меньше перепутаны предсказанные и истинные метки.
# Для регрессии функция потерь mae(mean_absolute_error) или mse(mean_squared_error) - являются средним(абсолютным или квадратичным) отклонением предсказанного от известных значений
# 3. metrics - то, что в процессе обучение подсчитывает для информации или для отбора лучшей модели(accuracy - это процент правильных ответов)

# обучим нейросеть
# model.fit(
#     X_train, y_train,
#     batch_size=1024,
#     epochs=30,
#     verbose=2  # выводить информацию по ходу дела, 1 - подробнее
# )

# callback (функция, вызываемая на каждой эпохе) ModelCheckpoint
from IPython.display import clear_output
# отнаследуемся от базового класса и переопределим конструктор и метод, вызываемый по окончанию эпохи
# class PlotLosses(keras.callbacks.Callback):
#     def __init__(self, metric=False, check_max=True):
#         super(PlotLosses, self).__init__()
#         self.logs = []
#         self.losses = []
#         self.val_losses = []
#         self.metric = metric or 'loss'
#         self.better = max if check_max else min
#
#     def on_epoch_end(self, epoch, logs={}):
#         clear_output(wait=True)
#         self.logs.append(logs)
#         x = range(1, len(self.logs) + 1)
#         self.losses.append(logs.get(self.metric))
#         if logs.get('val_' + self.metric):
#             self.val_losses.append(logs.get('val_' + self.metric))
#         if len(self.val_losses):
#             self.best_step = 1 + (
#                     self.val_losses.index(self.better(self.val_losses)) or 0)
#         else:
#             self.best_step = epoch
#         plt.plot(x, self.losses, ls='--', c='#323232', label=self.metric)
#         if len(self.val_losses):
#             plt.plot(x, self.val_losses, ls='-', c='#323232', label="val_" + self.metric)
#         plt.title("Step %d, %.4f" % (
#             len(self.logs),
#             logs.get(self.metric) or 0
#         ) + (", validation %.4f (best %.4f at %d)" % (
#             logs.get('val_' + self.metric) or 0,
#             self.better(self.val_losses if len(self.val_losses) else [0]) or 0,
#             self.best_step
#         ) if logs.get('val_' + self.metric) else ''))
#         plt.legend(loc='best')
#         plt.show()

# Для отбора лучшей модели используется валидационное множество(итоговое качество всё так же проверяют на тестовом)
# keras.backend.clear_session()
# model = create_model(number_features=X_train.shape[1],number_classes=y_train.shape[1])
# четверть тренировочных оставим под валидацию
# validation = int(train * 0.25)
# model.fit(
#     X_train[validation:], y_train[validation:],
#     batch_size=1024,
#     epochs=100,
#     validation_data=(X_train[:validation], y_train[:validation]),
#     verbose=0,  # НЕ выводить информацию по ходу дела
#     callbacks=[
#         PlotLosses(metric='accuracy'),
#         keras.callbacks.ModelCheckpoint(
#             'models/covtypes.h5',
#             monitor='val_accuracy',
#             save_best_only=True)])

# загрузим лучшую отобранную по accuracy на валидации модель
# best_model = keras.models.load_model('models/covtypes.h5')
# 'loss %.2f, accuracy %.2f' % tuple(best_model.evaluate(X_test, y_test))


"""
2 Автоэнкодеры
"""
# датасет fashion mnist
train = pd.read_csv('fashion-mnist_train.csv')
test = pd.read_csv('fashion-mnist_test.csv')
train_X = np.array(train.iloc[:, 1:])
# print("x_train.s", train_X.shape)
train_y = np.array(train.iloc[:, 0])  # target values of training dataset
test_X = np.array(test.iloc[:, 1:])
test_y = np.array(test.iloc[:, 0])  # target values of testing dataset
fashion = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'}
# plt.title(fashion[test_y[9]])
# plt.imshow(test_X[9].reshape(28, 28), cmap="gray")
# plt.show()

# сверточная нейросеть, которая будет сворачивать изображение до вектора, а потом разворачивать в изображение обратно
# keras.backend.clear_session()
# def create_autoencoder(shape, vector_size=3):
#     input_layer = layer = keras.layers.Input(shape)
#     filters = 16
#     layer = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1))(layer)
#     # после этого у нас размерность данных 28 * 28 * filters
#     # возьмем максимум из получаемых данных максимум,
#     # padding same - означает дополнять изображение значениям краёв, когда окно выходит за его пределы
#     layer = keras.layers.MaxPool2D((2, 2), padding='same')(layer)
#     # после этого у нас размерность 14 * 14 * filters
#     # процедуру повторим
#     layer = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu')(layer)
#     layer = keras.layers.MaxPool2D((2, 2), padding='same')(layer)
#     # свернем всё в вектор
#     layer = keras.layers.Flatten()(layer)
#     encoded = keras.layers.Dense(vector_size, activation='relu')(layer)
#     encoder = keras.Model(input_layer, encoded, name='encoder')
#     # вернем всё обратно
#     decoder_input = keras.layers.Input((vector_size,))
#     layer = keras.layers.Dense(7 * 7 * filters, activation='relu')(decoder_input)
#     layer = keras.layers.Reshape((7, 7, filters))(layer)
#     layer = keras.layers.UpSampling2D((2, 2))(layer)
#     layer = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu')(layer)
#     layer = keras.layers.UpSampling2D((2, 2))(layer)
#     # реконструируем наше изображение будем выдавать степень "белизны"
#     output_layer = keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same')(layer)
#     decoder = keras.Model(decoder_input, output_layer, name='decoder')
#     model = keras.Model(input_layer, decoder(encoder(input_layer)))
#     model.compile('adam', 'mae')
#     return model, encoder, decoder
# autoencoder, encoder, decoder = create_autoencoder((28, 28, 1))  # 1 - это у нас один черно-белый канал
# print(encoder.summary())
# print(decoder.summary())

# в датасете градации серого от 0 до 255
# training_set = np.expand_dims(train_X, axis=-1) / 255.
# будем обучать только на пятой части датасета - для скорости
# поскольку у нас классификация - сделаем стратифицированный сплит
from sklearn.model_selection import train_test_split
# train_subset_X, _, train_subset_y, _ = train_test_split(
#     training_set, train_y,
#     random_state=1, test_size=0.8,
#     stratify=train_y)
# autoencoder.fit(
#     train_subset_X,
#     train_subset_X,  # да, здесь y = X, так как это AutoEncoder
#     epochs=100,
#     # batch_size=len(train_subset_X) // 30,  # большой пакет, 400
#     batch_size=None,  # большой пакет, 400
#     verbose=0,
#     callbacks=[PlotLosses()])

# автоэнкодер
# Автоэнкодеры полезны в случае чистки изображений от шумов(на вход при обучении подают зашумленные изображения, а как выход - чистые)
# figure, axes = plt.subplots(1, 2, figsize=(8, 4))
# axes[0].set_title(fashion[test_y[0]])
# axes[0].imshow(test_X[0], cmap='gray')
# axes[1].set_title(fashion[test_y[0]] + ' reconstructed')
# axes[1].imshow(
#     autoencoder.predict(
#         test_X[0].reshape(1, 28, 28, 1) / 255.
#     )[0].reshape(28, 28),
#     cmap='gray')
# figure, axes = plt.subplots(1, 2, figsize=(8, 4))
# np.random.seed(1)
# noised_sample = (test_X[0] / 255.).copy()
# for height in range(noised_sample.shape[0]):
#     for width in range(noised_sample.shape[1]):
#         if np.random.uniform() > 0.8:
#             noised_sample[height, width] += np.random.normal(0, 0.1)
# axes[0].set_title(fashion[test_y[0]] + ' with noise')
# axes[0].imshow(noised_sample, cmap='gray')
# axes[1].set_title(fashion[test_y[0]] + ' reconstructed')
# axes[1].imshow(
#     autoencoder.predict(
#         noised_sample.reshape(1, 28, 28, 1)
#     )[0].reshape(28, 28),
#     cmap='gray')

# более полезны тем, что есть трехмерное представление каждого изображения
# encoder.predict(noised_sample.reshape(1, 28, 28, 1))[0].tolist()

# что представляют собой полученные вектора на тестовом множестве(отобразить метки классов)
from mpl_toolkits.mplot3d import Axes3D
# axes = plt.subplot(projection='3d')
# test_vectors = encoder.predict(test_X.reshape(-1, 28, 28, 1) / 255.)
# axes.scatter(
#     test_vectors[:, 0],
#     test_vectors[:, 1],
#     test_vectors[:, 2],
#     c=test_y,
#     cmap='jet')
from matplotlib.patches import Rectangle
# figure, axis = plt.subplots(10, 1, figsize=(2, 5))
# for color in range(10):
#     axis[color].add_patch(Rectangle((0, 0), 1, 2, alpha=1, facecolor=plt.get_cmap('jet')(color / 10)))
#     axis[color].axis('off')
#     axis[color].annotate(fashion[color], (0.5, 0.5), c='white', ha='center', va='center')
# plt.show()

# либо кластеризовывать, либо строить классификатор
# train_vectors = encoder.predict(train_X.reshape(-1, 28, 28, 1) / 255.)
# axes = plt.subplot(projection='3d')
# axes.scatter(
#     train_vectors[:, 0],
#     train_vectors[:, 1],
#     train_vectors[:, 2],
#     c=train_y,
#     cmap='jet')

# from hdbscan import HDBSCAN  # библиотека быстрой кластеризации
from sklearn.metrics import silhouette_score
# best_score = -np.inf
# best_number = 1
# best_size = 0
# for min_size in [50, 100, 200]:
    # clusterer = HDBSCAN(min_cluster_size=min_size).fit(test_vectors)
    # if len(pd.unique(clusterer.labels_)) < 2:
    #     continue
    # score = silhouette_score(test_vectors, clusterer.labels_)
    # if score > best_score:
    #     best_score = score
    #     best_size = min_size
    #     best_number = len(pd.unique(clusterer.labels_))
# 'best clusters number %d with min cluster size = %d and score %.2f' % (best_number, best_size, best_score)

# разложилось всё не по 10 классам
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# classifier = LogisticRegression(random_state=1, max_iter=1000).fit(train_vectors, train_y)
# knn = KNeighborsClassifier(weights='distance').fit(train_vectors, train_y)
# "logreg acc %.2f, knn acc %.2f" % (classifier.score(test_vectors, test_y), knn.score(test_vectors, test_y))

# показать классификатору новые картинки
from PIL import Image
# test_1 = Image.open('media/test_1_square.jpg').convert('L')  # grayscale
# test_1 = 1. - np.array(test_1.resize((28, 28))) / 255.
# test_2 = Image.open('media/test_2_square.jpg').convert('L')
# test_2 = 1. - np.array(test_2.resize((28, 28))) / 255.
# plt.subplot(1, 2, 1)
# plt.imshow(test_1, cmap='gray')
# plt.subplot(1, 2, 2)
# plt.imshow(test_2, cmap='gray')
# fashion[classifier.predict(encoder.predict(test_1.reshape(1, 28, 28, 1)))[0]], \
# fashion[classifier.predict(encoder.predict(test_2.reshape(1, 28, 28, 1)))[0]]


"""
3 Классификация и сегментация изображений
"""
# test_3 = Image.open('test_3.jpg').convert('L')
# test_3 = 1. - np.array(test_3.resize((28, 28))) / 255.
# keras.backend.clear_session()
# def create_classifier(shape, number_classes):
#     model = keras.Sequential([
#         keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu', input_shape=shape),
#         keras.layers.MaxPool2D((2, 2), padding='same'),
#         keras.layers.Dropout(0.1),
#         keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
#         keras.layers.MaxPool2D((2, 2), padding='same'),
#         keras.layers.Dropout(0.1),
#         keras.layers.Flatten(),
#         keras.layers.Dense(number_classes, activation='softmax')])
#     model.compile('adam', 'categorical_crossentropy', metrics=['acc'])
#     return model
# classifier = create_classifier((28, 28, 1), 10)
# print(classifier.summary())
# classifier.fit(
#     training_set,
#     keras.utils.to_categorical(train_y),
#     batch_size=500,
#     epochs=30,
#     verbose=0,
#     callbacks=[PlotLosses(metric='acc')])
# 'loss %.2f, accuracy %.2f' % tuple(classifier.evaluate(test_X.reshape(-1, 28, 28, 1), pd.get_dummies(test_y)))

from tf_keras_vis.gradcam import GradcamPlusPlus
from tf_keras_vis.utils import normalize
# подменим активацию последнего слоя
# def model_modifier(model):
#     model.layers[-1].activation = keras.activations.linear
#     return model
# tf-keras-vis способ gradcam - отображаются те пиксели входного изображения, которые привели классификатор к конкретному ответу
# gradcam = GradcamPlusPlus(
#     classifier,
#     model_modifier,
#     clone=False)
# cam = gradcam(
#     lambda output: output[0],  # для этой библиотеки это откуда как брать loss
#     test_3.reshape(1, 28, 28, 1),
#     penultimate_layer=-1  # заберем последний слой
# )
# cam = normalize(cam)
# heatmap = np.uint8(plt.cm.jet(cam[0])[..., :3] * 255)
# plt.title(fashion[np.argmax(classifier.predict(test_3.reshape(1, 28, 28, 1))[0])])
# plt.imshow(test_3.reshape(28, 28), cmap='gray')
# plt.imshow(heatmap, cmap='jet', alpha=0.5)
# plt.tight_layout()
# plt.show()


"""
4 Готовые нейросети для работы с изображениями
"""
# from keras.applications import NASNetMobile
# from keras.applications.nasnet import preprocess_input, decode_predictions
# nasnet = NASNetMobile(weights="imagenet")
# cat_image = np.array(Image.open('media/white_cat.jpg').resize((224, 224)))
# cat_image = preprocess_input(np.asarray([cat_image]))
# cat_image_prediction = nasnet.predict(cat_image)
# print("Class, Description, Probability")
# for cat_prediction in decode_predictions(cat_image_prediction, top=5)[0]:
#     print(cat_prediction)


"""
5 Тексты и нейросети
"""
# Google BERT очень большого размера, и был обучен на текстах Википедии решать две задачи:
# 1. Предсказывать пропущенные слова в тексте: “я пошел в ? и купил ?” - “я пошел в магазин и купил молоко”
# 2. Предсказывать, является ли текст продолжением некоторого начала: “я пошел в магазин – и купил молоко”(ok) и “я пошел в магазин – и пингвины не летают”(fail)
# from deeppavlov.core.common.file import read_json
# bert_config = read_json(configs.embedder.bert_embedder)
# bert_config['metadata']['variables']['BERT_PATH'] = 'd:/workspace/bert/bert-base-multilingual-cased/'
# bert = build_model(bert_config)
# question_answers = [
#     ('Кто первым полетел в космос?', 'Юрий Гагарин'),
#     ('Кто был первым президентом России?', 'Михаил Горбачев'),
#     ('Зачем автомобилю руль?', 'Чтобы водитель мог поворачивать')]
# qa_vectors = [
#     bert(question[0])[5][0].tolist() for question in question_answers]
# from sklearn.neighbors import NearestNeighbors
# knowledge_base = NearestNeighbors(n_neighbors=1, metric='cosine').fit(qa_vectors)
# question = 'Кто был первым космонавтом?'
# answer_index = knowledge_base.kneighbors(
#     [bert(question)[5][0]], return_distance=False)[0][0]
# print(question, question_answers[answer_index][1])


"""
6 Рекуррентные нейросети для временных рядов
"""
