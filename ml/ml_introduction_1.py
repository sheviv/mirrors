# https://timeseries-ru.github.io/course/day_1.html
# Курс “Введение в анализ данных и машинное обучение”

# Определения модели, алгоритма, метрик, задач машинного обучения
# "лементы статистики (случайная величина, выборочные статистики, распределение, пара-тройка теорем)
# Почему все это работает: теория Валианта и vc-размерность в картинках.
# Почему и как python и Jupiter lab: основы. Переменные, списки и словари, функции, классы и объекты
# Цикл моделирования crisp-dm. Валидация и кроссвалидация. End2End-пример
# Обзор разделов всего курса на примерах предсказаний

# импорт библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# другой вариант импорта
from sklearn.cluster import KMeans
# вариант импорта из недр библиотеки
import sklearn.datasets as datasets
# импортируем алгоритм, строящий линейные модели
from sklearn.linear_model import LinearRegression
# импортируем алгоритм решающего дерева
from sklearn.tree import DecisionTreeRegressor
# импортируем алгоритм ближайших соседей
from sklearn.neighbors import KNeighborsRegressor
# импортируем подсчет средней абсолютной ошибки
from sklearn.metrics import mean_absolute_error

# height = [
#     np.random.normal(170, 5) for _ in range(300)  # синтетические данные
# ]
# weight = [
#     sample - 110 - np.random.normal(0, 1) for sample in height
# ]
# results = []
# estimator_names = [
#     'linear', 'deep tree', 'shallow tree', 'neighbors'
# ]
# for number_points in range(5, 31):
#     errors = {
#         name: [] for name in estimator_names
#     }
#     for index in range(100):
#         random_indices = np.random.randint(
#             0, len(weight),
#             size=number_points
#         )
#         for iterator, estimator in enumerate([
#             LinearRegression(),
#             DecisionTreeRegressor(random_state=1),
#             DecisionTreeRegressor(max_depth=3, random_state=1),
#             KNeighborsRegressor(weights='distance')
#         ]):
#             yhat = estimator.fit(
#                 np.array(height).reshape(-1, 1)[random_indices],
#                 np.array(weight)[random_indices]
#             ).predict(
#                 np.array(height).reshape(-1, 1)
#             )
#             errors[estimator_names[iterator]].append(
#                 mean_absolute_error(weight, yhat)
#             )
#     percents = {}
#     for name in estimator_names:
#         percents[name] = sum([
#             1 if error > 1. else 0 for error in errors[name]
#         ])
#         results.append([
#             number_points,
#             percents[name],
#             np.mean(errors[name]),
#             name
#         ])
#     print("""Шаг %d процент "плохих" моделей: линейный алгоритм: %.f, глубокое дерево: %.f, неглубокое дерево %.f, соседи %.f""" % (
#         number_points,
#         percents['linear'],
#         percents['deep tree'],
#         percents['shallow tree'],
#         percents['neighbors']
#     ))
# sns.scatterplot(x="points", y="error", size="percent", hue="type", data=pd.DataFrame(results, columns=[
#     'points', 'percent', 'error', 'type'
# ]))
# plt.subplot(1, 2, 1)
# plt.title('Достаточно одной линии')
# plt.scatter([1, 2, 3], [1, 3, 2])
# plt.plot([1, 3], [2, 1], c='red', ls='--')
# plt.plot([1, 3], [2, 3], c='red', ls='--')
# plt.plot([3, 2], [3, 1], c='red', ls='--')
# plt.subplot(1, 2, 2)
# plt.title('Нужно две линии')
# plt.scatter([1, 2, 3, 4], [1, 3, 2, 4])
# plt.plot([2, 1], [1, 2], c='red', ls='--')
# plt.plot([4, 3], [1, 4], c='red', ls='--')
# plt.annotate("Две неразделимые\nодной линией части", (1, 2.5))

# загрузим датасет из библиотеки
# X, y = datasets.load_iris(return_X_y=True)
# KMeans - это класс с методами
# clusterer = KMeans(n_clusters=3).fit(X)
# запись X[:, 1] - означает взять ВТОРУЮ колонку массива
# sns.scatterplot(x=X[:,1], y=X[:,2], hue=clusterer.labels_)
# plt.show()

# Цикл моделирования и кросс-валидация
# 1 Понимание задачи и целей
# цель - создать модель, которая по признакам цветка определяла бы их сорт.
# Отбор моделей необходимо производить с учетом того, что сорта стоят по-разному!

# 2 Разведочный анализ
from sklearn.datasets import load_iris
# распечатаем какие есть признаки у данных
iris = load_iris()
ir = iris.feature_names
# [print(i) for i in ir]
# загрузим данные в pandas-таблицу
features_data = {
    iris.feature_names[index]: iris.data[:, index] \
    for index in range(len(iris.feature_names))}
features_data['sort'] = iris.target
features_data['name'] = [iris.target_names[sort] for sort in iris.target]
frame = pd.DataFrame(
    features_data,
    columns=iris.feature_names + ['sort', 'name'])
# и посмотрим на случайную выборку из 10 цветов
# print(frame.sample(10))
# print(frame.describe(include='all'))
# print(frame['name'].value_counts())
# Посмотрим на данные визуально, причем попарно
features = iris.feature_names
sns.pairplot(data=frame[features + ['name']], hue="name")
# plt.show()

# 3 Подготовка данных
# признак sepal width (cm) - уберем, так как по этому признаку цветы очень схожи
reduced_features = list(features)  # скопируем
reduced_features.remove('sepal width (cm)')
# Разобъем датасет на тренировочную и тестовую части. Пусть тестовая часть будет 25% от всего датасета.
from sklearn.model_selection import train_test_split
train, test = train_test_split(
    frame,
    random_state=1,
    test_size=0.25,
    stratify=frame.sort  # об этом чуть ниже
)

# 4 Моделирование
# модель классификации на основе ближайших соседей.
from sklearn.neighbors import KNeighborsClassifier
# создадим объект-классификатор и будем определять класс по 5 ближайшим соседям
classifier_5 = KNeighborsClassifier(n_neighbors=5)
# создадим второй, и будем определять уже по 10
classifier_10 = KNeighborsClassifier(n_neighbors=10)
# применим алгоритм к тренировочным данным
classifier_5.fit(train[reduced_features], train.sort)
classifier_10.fit(train[reduced_features], train.sort)
# получим предсказания на тренировочном и тестовом множествах
yhat_train_5 = classifier_5.predict(train[reduced_features])
yhat_test_5 = classifier_5.predict(test[reduced_features])
yhat_train_10 = classifier_10.predict(train[reduced_features])
yhat_test_10 = classifier_10.predict(test[reduced_features])

# 5 Проверка качества
# в качестве метрики - процент правильно отмеченных цветов
from sklearn.metrics import accuracy_score
# print('classifier_5', 'train', "%.2f" % accuracy_score(train.sort, yhat_train_5))
# print('classifier_5', 'test', "%.2f" % accuracy_score(test.sort, yhat_test_5))
# print('classifier_10', 'train', "%.2f" % accuracy_score(train.sort, yhat_train_10))
# print('classifier_10', 'test', "%.2f" % accuracy_score(test.sort, yhat_test_10))
#  измерим ошибку в деньгах
# следующая функция подсчитывает таблицу ошибок по классам
prices = {'setosa': 100,
          'versicolor': 120,
          'virginica': 200}
def money_error(true_set, predictions):
    cost_true = sum([prices[true_set['name'][index]] for index in true_set.index])
    cost_predicted = sum([prices[iris.target_names[int(prediction)]] for prediction in predictions])
    # print("Должно стоить %d, а предсказано %d" % (cost_true, cost_predicted))
money_error(test, yhat_test_5)
money_error(test, yhat_test_10)
# как алгоритмы ошибаются
from sklearn.metrics import confusion_matrix
# На главной диагонали матрицы стоит число правильно классифицированных примеров, в остальных ячейках - перепутанные
# при предсказании метки(по горизонтали - актуальные значения,по вертикали - предсказанные)
# print(confusion_matrix(test.sort, yhat_test_5))
# print(confusion_matrix(test.sort, yhat_test_10))
# проверим со качество всеми признаками (на 10 соседях)
classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(train[features], train.sort)
yhat_test = classifier.predict(test[features])
# print(confusion_matrix(test.sort, yhat_test))
money_error(test, yhat_test)
# кросс-валидация
from sklearn.model_selection import cross_val_score
metrics = cross_val_score(
    KNeighborsClassifier(n_neighbors=3),
    X=frame[reduced_features],
    y=frame.sort,
    cv=5  # пять частей
)
# print("Ожидаемое среднее качество %.2f" % metrics.mean())
# print("Разброс среднего качества %.2f" % metrics.std())
