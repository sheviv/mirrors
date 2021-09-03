# https://timeseries-ru.github.io/course/day_2.html
# Курс “Введение в анализ данных и машинное обучение”

# библиотеки работы с данными numpy, pandas, pyplot, seaborn, ipywidgets,
# понятие разведочного анализа и предобработки признаков. поймем, почему нельзя полагаться только на выборочные статистики как числа,
# линейные модели и градиентный спуск,
# деревья решений и ансамбли моделей на основе деревьев решений,
# алгоритм ближайших соседей и соседство как таковое (весьма важно на практике), в том числе на примерах текстовых данных,
# (пока) поверхностно: нейронные сети в библиотеке scikit-learn.

import numpy as np


"""
# 1 Библиотеки работы с данными
"""
# запретим предупреждения
import warnings

warnings.filterwarnings('ignore')
# подключим pandas с общепринятым сокращением
import pandas as pd

# dataframe = pd.read_csv(
#     'abalone.csv',
#     sep=',',  # что является разделителем колонок в файле,
#     decimal='.',  # что является разделителем десятичных дробей в записи чисел
#     parse_dates=[],  # мы знаем, что дат у нас нет, если бы они были, здесь можно было бы перечислить колонки
#     header=0  # названия колонок в первой строке
# )
# посмотрим на случайные 15 записей
# print(dataframe.sample(15))

# замена имени Class_number_of_rings на Rings
# dataframe = dataframe.rename(columns={"Class_number_of_rings": "rings"})
# dataframe = dataframe.rename(columns=str.lower)
# dataframe.columns = dataframe.columns.str.replace('_', ' ')

# описательные статистики для числовых полей
# print(dataframe.describe())

#  количество ракушек разных полов
# print(dataframe.groupby('Sex')['Rings'].count())

# количество ракушек по полам, в случае если диаметр ракушки больше среднего
# dataframe[dataframe.diameter > dataframe.diameter.mean()].groupby('sex')['rings'].count()  # маленький диаметр - это преимущественно дети

# срез данных по колонкам и строкам, первая и последние колонки и последние три строки
# cv = dataframe[['sex', 'rings']][-3:]

# сортировка датасета - и посмотрим максимальный
# .values - это получить колонку как numpy-массив
# cv = dataframe.sort_values('rings')["rings"].values[-1]  # ракушка-старожил

# каждый датафрейм (объект таблицы) имеет индекс
# dataframe.index[:4:2]

# получить записи по индексу таблицы, и выборочно колонки
# cv = dataframe.loc[dataframe.index[:4:2]][dataframe.columns[:3]]

import matplotlib.pyplot as plt  # тоже общепринятое сокращение
# распределение колец ракушек
import seaborn as sns
# plt.title("Распределение колец по количеству.\nЧем-то похоже на нормальное, но точно не оно")
# sns.distplot(dataframe.rings)

# попарные точечных диаграммы в зависимости какой либо колонки
# sns.pairplot(data=dataframe[['length', 'diameter', 'height', 'whole weight', 'rings', 'sex']], hue='sex')

# диаграмма серий (пример с датами)
# airpassengers = pd.read_csv('data/airpassengers.csv', parse_dates=['Month'])
# sns.lineplot(x="Month", y="#Passengers", data=airpassengers)


"""
2 Разведочный анализ(понимание данных) и препроцессинг
"""
# квартет Энскомба
# quartet = pd.read_csv('anscombes.csv').drop('id', axis='columns')
#
# plt.figure(figsize=(10, 10))
# for index, dataset in enumerate(['I', 'II', 'III', 'IV']):
#     plt.subplot(2, 2, index + 1)
#     plt.scatter(
#         quartet[quartet.dataset == dataset].x,
#         quartet[quartet.dataset == dataset].y,
#         c=['blue', 'green', 'red', 'black'][index]
#     )
#     plt.xlim(0, 20)
#     plt.ylim(0, 20)
# plt.show()

# Это четыре разных датасета, но посмотрим их статистики.
# cv = quartet.groupby('dataset').describe().T  # количество, среднее и разброс у всех наборов одинаковый

# смотреть попарные диаграмми глазами
from sklearn.linear_model import LinearRegression
# coefs = []
# plt.figure(figsize=(10, 10))
# for index, dataset in enumerate(['I', 'II', 'III', 'IV']):
#     plt.subplot(2, 2, index + 1)
#     plt.scatter(
#         quartet[quartet.dataset == dataset].x,
#         quartet[quartet.dataset == dataset].y,
#         c=['blue', 'green', 'red', 'black'][index])
#     model = LinearRegression().fit(
#         quartet[quartet.dataset == dataset].x.values.reshape(-1, 1),
#         quartet[quartet.dataset == dataset].y)  # y = kx + b
#     coefs.append([
#         model.coef_[0],  # это k
#         model.intercept_  # это b
#     ])
#     line = model.predict(quartet[quartet.dataset == dataset].x.values.reshape(-1, 1))
#     plt.plot(
#         quartet[quartet.dataset == dataset].x,
#         line,
#         ls='-.',
#         c='orange',
#         lw=1)
#     plt.xlim(0, 20)
#     plt.ylim(0, 20)
# df = np.array(coefs)[:, 0], np.array(coefs)[:, 1]  # Коэффициенты практически одинаковые.

# sns.pairplot(dataframe[['whole weight', 'diameter', 'sex']], hue='sex')  # зависимость между весом ракушки и её диаметром
# степенная? можем предположить, что извлечение корня “исправит ситуацию”, и может превратить зависимость в линейную

# dataframe_copy = dataframe.copy()
# dataframe_copy['root_weight'] = np.sqrt(dataframe_copy['whole weight'])
# sns.pairplot(dataframe_copy[['root_weight', 'diameter', 'sex']], hue='sex')  # слегка “выпрямили” зависимость
# степень еще меньше одной второй

# попробуем с третьей степенью
# dataframe_copy['changed_weight'] = np.power(dataframe_copy['whole weight'], 1 / 3)
# sns.pairplot(dataframe_copy[['changed_weight', 'diameter', 'sex']], hue='sex')  # угадали

# категориальный признак
# мужскому полу сопоставить вектор (1, 0, 0), женскому (0, 1, 0), а детскому соответственно (0, 0, 1)
# всё это можно сделать функцией pandas get_dummies
# processed_data = pd.get_dummies(dataframe['sex'])

# чтобы две таблицы склеить по индексу, можно использовать такой код
# number_data = processed_data.join(dataframe).drop('sex', axis='columns')
# cv = number_data.head()


"""
3 Линейные модели и градиентный спуск
"""
# def update(x, y, k, b, alpha):
#     number = len(y)
#     # подсчитаем производные по коэффициентам
#     change_k = -2 * sum([x[index] * (y[index] - k * x[index] - b) for index in range(number)])
#     change_b = -2 * sum([x[index] * k - y[index] - b for index in range(number)])
#     # параметр alpha - называется скорость обучения
#     new_k = k - alpha * change_k / number
#     new_b = b - alpha * change_b / number
#     return new_k, new_b
#
# def train_linear(x, y, alpha=0.1, epochs=50):
#     # инициализируем случайно наши коэффициенты модели
#     k, b = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
#     for epoch in range(epochs):
#         k, b = update(x, y, k, b, alpha)
#         if epoch % 10 == 0 and epoch > 0:  # каждый десятый шаг
#             pass
#             # print("%0d: среднее абсолютное отклонение численно составляет %.4f" % (epoch,np.mean(np.abs(y - k * x - b))))
#     return k, b
from sklearn.utils import shuffle
# random_indices = shuffle(range(len(dataframe)), random_state=1)
# train_indices = random_indices[:-300]  # 300 точек - будет тест
# test_indices = random_indices[-300:]
# k, b = train_linear(
#     dataframe['length'].values[train_indices],
#     dataframe['diameter'].values[train_indices])
# plt.title('Результаты модели')
# plt.scatter(dataframe['length'], dataframe['diameter'])
# plt.plot(dataframe['length'], k * dataframe['length'] + b, c='orange')
# plt.annotate("Запомним эту точку под\nназванием \"выброс\"", (0.185, 0.38), (0.2, 0.5), arrowprops={"arrowstyle": '->'})

# Выбросы - это редкие точки, которые сильно отличаются от поведения всех остальных
# “плотность” точек очень высокая, и влияние выбросов небольшое. Для иллюстрации, возьмем только вторую четверть датасета
# sliced = slice(len(train_indices) // 4, len(train_indices) // 2)
# plt.scatter(dataframe['length'][sliced], dataframe['diameter'][sliced])

# Результаты моделей на тестовом множестве
from sklearn.linear_model import LinearRegression, Ridge
# linear_model = LinearRegression().fit(
#     dataframe.length.values[train_indices][sliced].reshape(-1, 1),
#     dataframe.diameter[train_indices][sliced].values)
# ridge_model = Ridge(
#     # это настройка штрафа, чем она выше, тем сильнее регуляризация
#     alpha=1.25).fit(
#     dataframe.length.values[train_indices][sliced].reshape(-1, 1),
#     dataframe.diameter[train_indices][sliced].values)
# k, b = train_linear(
#     dataframe['length'].values[train_indices][sliced],
#     dataframe['diameter'].values[train_indices][sliced])
# # а предскажем - на тестовом
# linear_predictions = linear_model.predict(
#     dataframe.length.values[test_indices].reshape(-1, 1))
# ridge_predictions = ridge_model.predict(
#     dataframe.length.values[test_indices].reshape(-1, 1))
# plt.title('Результаты моделей на тестовом множестве')
# plt.scatter(dataframe['length'].values[test_indices], dataframe['diameter'].values[test_indices], alpha=0.1)
# plt.plot(dataframe['length'].values[test_indices], k * dataframe['length'].values[test_indices] + b, ls='--', c='orange', label="Наивная")
# plt.plot(dataframe['length'].values[test_indices], linear_predictions, c='green', ls='--', label="Linear")
# plt.plot(dataframe['length'].values[test_indices], ridge_predictions, c='red', ls='--', label="Ridge")
# plt.legend(loc='best')
# plt.show()

# проверим метрику R-квадрат - самая распространенная метрика для задачи регрессии.
# всегда меньше либо равна единице, чем ближе к единице - тем лучше предсказания модели совпадают с истинными значениями
from sklearn.metrics import r2_score
# print("Наивная модель на тестовом множестве %.5f, Linear и Ridge (там же): %.5f, %.5f" % (
#     r2_score(dataframe.diameter.values[test_indices], k * dataframe['length'].values[test_indices] + b),
#     r2_score(dataframe.diameter.values[test_indices], linear_predictions),
#     r2_score(dataframe.diameter.values[test_indices], ridge_predictions)))

# задача классификации с помощью линейной модели - LogisticRegression(sigmoid)
# применить логистическую регрессию к датасету Iris.
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# X, y = load_iris(return_X_y=True)
# оставим только два последних признака
# X = X[:, -2:]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True)
# создадим модель
# logreg = LogisticRegression(random_state=1, multi_class='auto').fit(X_train, y_train)
from sklearn.metrics import accuracy_score  # метрика - количество правильных ответов
# print("Точность классификации на %d тестовых примерах %.3f" % (
#     len(y_test),
#     accuracy_score(
#         y_test,
#         logreg.predict(X_test))))

# Матрица несоответствий на тестовом множестве
from sklearn.metrics import confusion_matrix
# plt.title("Матрица несоответствий на тестовом множестве")
# sns.heatmap(confusion_matrix(y_test, logreg.predict(X_test)), annot=True)

# как разделились классы на плоскости(Границы принятия решений)
# def plot_decisions(x, y, targets, classifier, labels=None):
#     x_min, x_max = x.min() - 1, x.max() + 1
#     y_min, y_max = y.min() - 1, y.max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
#                          np.arange(y_min, y_max, 0.01))
#     z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
#     z = z.reshape(xx.shape)
#     c = plt.contourf(xx, yy, z, cmap='Paired', alpha=0.3)
#     for index in range(len(pd.unique(targets))):
#         indices = np.where(targets == index)
#         plt.scatter(x[indices], y[indices], color=[
#             'b', 'r', 'y'
#         ][index], label=labels[index] if labels is not None else index)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
# plt.title('Границы принятия решений')
# plot_decisions(X[:, 0],X[:, 1],y,logreg)


"""
4 Деревья решений и ансамбли моделей на их основе
"""
# сли метки разбиты как 50/50 - энтропия 1, если как 0/100 (есть только одна метка) - энтропия 0
# одиночное дерево на ирисе
from sklearn.tree import DecisionTreeClassifier, plot_tree
# деревья by design умеют в мультикласс
# decision_tree = DecisionTreeClassifier(random_state=1).fit(X_train, y_train).fit(X_train, y_train)
# plt.title("Матрица несоответствий на тестовом множестве")
# sns.heatmap(confusion_matrix(y_test, decision_tree.predict(X_test)), annot=True)
# plt.title('Как дерево подобрало пороги')
# Одиночные деревья - всегда переобучаются
# plot_decisions(X[:, 0], X[:, 1], y, decision_tree)  # “слишком глубоко” погрузилось в данные

# обрежем по глубине
# decision_tree_shallow = DecisionTreeClassifier(
#     criterion='entropy',
#     max_depth=3,  # максимум 3 вопроса
#     random_state=1).fit(X_train, y_train)
# plt.title('Другая граница решений')
# plot_decisions(X[:, 0], X[:,1], y, decision_tree_shallow)

# отрисовать дерево решений в виде графа
# plt.figure(figsize=(10, 10))
# plot_tree(decision_tree_shallow, filled=True)

# Градиентный бустинг над решающими деревьями and Случайный лес решающих деревьев
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# boosting = GradientBoostingClassifier(
#     n_estimators=10,  # количество деревьев
#     max_depth=5,  # глубина деревьев
#     random_state=1).fit(X_train, y_train)
# forest = RandomForestClassifier(
#     n_estimators=10,
#     random_state=1).fit(X_train, y_train)

# plt.title("Матрица несоответствий на тестовом множестве для градиентного бустинга")
# sns.heatmap(confusion_matrix(y_test, boosting.predict(X_test)), annot=True)
# plt.title('Граница решений для градиентного бустинга')
# plot_decisions(
#     X[:,0],
#     X[:,1],
#     y,
#     boosting)

# plt.title("Матрица несоответствий на тестовом множестве для случайного леса")
# sns.heatmap(confusion_matrix(y_test, forest.predict(X_test)), annot=True)
# plt.title('Граница решений для случайного леса')
# plot_decisions(
#     X[:,0],
#     X[:,1],
#     y,
#     forest)


"""
5 Алгоритм ближайших соседей (и немного о текстах)
"""
# пример на датасете Iris
from sklearn.neighbors import KNeighborsClassifier
from ipywidgets import interact
# обучим на всём датасете
# knn_classifier = KNeighborsClassifier(n_neighbors=3).fit(X, y)
# def plot_point(point_x, point_y):
#     sort = knn_classifier.predict([[point_x, point_y]])[0]
#     plt.title("Предсказанный класс %d" % sort)
#     plt.scatter(X[:, 0].tolist() + [point_x],X[:, 1].tolist() + [point_y],c=y.tolist() + [sort])
#     plt.annotate(
#         "Наша точка",
#         (point_x, point_y),
#         (point_x + 0.25, point_y + 0.25),
#         arrowprops={"arrowstyle": '->'})
# interact(plot_point,point_x=(0, 7.5, 0.2),point_y=(0, 3.5, 0.2))
# plt.title('Граница решений для ближайших соседей')
# plot_decisions(X[:, 0],X[:, 1],y,KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train))
# plt.show()

# texts = ["Не работает интернет!", "Не работает телевидение", "Проблемы с телевидением", "Интернет сломался"]
# Импортируем один из самых простых "векторизаторов" текстов он укладывает вектора на единичную сферу
from sklearn.feature_extraction.text import HashingVectorizer
# vectorizer = HashingVectorizer(n_features=3).fit(texts)
# space = vectorizer.transform(texts)
# print("Вектора текстов", space.todense())
from sklearn.neighbors import NearestNeighbors
# инстанцируем объект, который будет находить ближайшего соседа
# nearest = NearestNeighbors(n_neighbors=1).fit(space)
# text = "Что-то не так с интернетом"
# index = nearest.kneighbors(vectorizer.transform([text]), return_distance=False)[0][0]
# print("Ближайший сосед %s: " % text, texts[index])
# text = "Что за ерунда с ТВ!"
# index = nearest.kneighbors(vectorizer.transform([text]), return_distance=False)[0][0]
# print("Ближайший сосед %s: " % text, texts[index])


"""
6 Нейросети в scikit-learn
"""
# регрессоры и классификаторы с помощью нейронных сетей
# Нейросети хорошо работают с маленькими числами, так как большие числа дают большие значения производных.
# два способа подготовки данных для нейросети: standard scaling и minmax scaling
data = pd.read_csv('abalone.csv')
data = data.rename(columns={"Class_number_of_rings": "rings"})
data = data.rename(columns=str.lower)
data.columns = data.columns.str.replace('_', ' ')
data = pd.get_dummies(data.sex).join(data).drop(['sex'], axis='columns')
features = list(data.columns)
features.remove('rings')

# задача предсказания количества колец
# качество оценивать по кросс-валидации
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier

# standard scaler - вычитает среднее и делит на разброс
changed = list(features)
for change in ['F', 'M', 'I']:
    changed.remove(change)
# при преобразовании пропустим признаки пола
shuffled_X, shuffled_y = shuffle(
    np.hstack([data[['F', 'M', 'I']].values.reshape(-1, 3),
               StandardScaler().fit_transform(data[changed])]),
    data['rings'].values + 1.5,  # перейдем сразу к возрасту
    random_state=1)
target_scaler = StandardScaler()
scaled_target = target_scaler.fit_transform(shuffled_y.reshape(-1, 1))
regressor = MLPRegressor(
    hidden_layer_sizes=[20, 20],  # два скрытых слоя на 20 нейронов
    activation='relu',
    max_iter=1000,  # сколько итераций подгонки
    random_state=1  # для воспроизводимости
)
scores = cross_val_score(
    regressor,
    X=shuffled_X,
    y=scaled_target.flatten(),  # сделаем массив "плоским"
    cv=3,
    scoring='r2')
# "R2 mean: %.3f, std %.3f" % (scores.mean(), scores.std())  # Не большой R2 - выбросы

# почистим данные от них
# sns.pairplot(data=pd.read_csv('abalone.csv')[['height', 'diameter', 'sex']], hue='sex')

bad_indices = shuffle(np.where(data.height > 0.4), random_state=1)[0]
good_indices = list(range(len(data)))
for index in bad_indices:
    good_indices.remove(index)
# print(bad_indices)

# Выкинем их из данных и попробуем еще раз
regressor = MLPRegressor(
    hidden_layer_sizes=[20, 20],  # два скрытых слоя на 20 нейронов
    activation='relu',
    max_iter=1000,  # сколько итераций подгонки
    random_state=1  # для воспроизводимости
)
scores = cross_val_score(
    regressor,
    X=shuffled_X[good_indices],
    y=scaled_target.flatten()[good_indices],  # сделаем массив "плоским"
    cv=3,
    scoring='r2')
# "R2 mean: %.3f, std %.3f" % (scores.mean(), scores.std())  # Метрика улучшилась

shuffled_X = shuffled_X[good_indices]
shuffled_y = shuffled_y[good_indices]
# plt.title('Распределение возраста, все данные')
# sns.distplot(shuffled_y)

# Трансформированная целевая величина
train = int(len(shuffled_y) * 0.8)
test = len(shuffled_y) - train
# PowerTransform - подбирает для одной величины степенное преобразование так, чтобы её распределение было  более близко к нормальному
from sklearn.preprocessing import PowerTransformer
target_processor = PowerTransformer().fit(shuffled_y[:train].reshape(-1, 1))
transformed_y = target_processor.transform(shuffled_y.reshape(-1, 1)).flatten()
plt.title('Трансформированная целевая величина train/test')
sns.distplot(transformed_y[:train])
sns.distplot(transformed_y[train:])

from sklearn.metrics import mean_absolute_error
regressor = MLPRegressor(
    hidden_layer_sizes=[20, 20],
    activation='relu',
    max_iter=1000,
    random_state=1)
regressor.fit(shuffled_X[:train], transformed_y[:train])
"R2 %.3f, ошибка в возрасте: %.2f, разброс значений возраста %.2f" % (
    r2_score(
        shuffled_y[train:],
        target_processor.inverse_transform(
            regressor.predict(shuffled_X[train:]).reshape(-1, 1)
        )
    ),
    mean_absolute_error(
        shuffled_y[train:],
        target_processor.inverse_transform(regressor.predict(shuffled_X[train:]).reshape(-1, 1))),
    shuffled_y[train:].std())

# Посмотрим теперь классификацию на датасете Iris
# classifier = MLPClassifier(
#     hidden_layer_sizes=[32, 12],
#     activation='tanh',
#     max_iter=1000,
#     random_state=1)
# X_changed = MinMaxScaler(
#     feature_range=(-1, 1)
# ).fit_transform(X)
# scores = cross_val_score(
#     classifier,
#     X=X_changed,
#     y=y,
#     cv=3,
#     scoring='accuracy')

# нейросетью можно разграничить классы датасета Iris
# iris_X, iris_y = shuffle(X_changed, y, random_state=1)
# classifier.fit(iris_X[:-50], iris_y[:-50])
# plt.title("Обратите внимание - оси отмасштабированы!")
# plot_decisions(X_changed[:, 0], X_changed[:, 1], y, classifier)
