# https://timeseries-ru.github.io/course/day_3.html
# Курс “Введение в анализ данных и машинное обучение”

# Немного про отбор признаков и про поиск гиперпараметров,
# Задача понижения размерности данных,
# Кластеризация,
# Поиск аномалий в данных,
# Одномерные временные ряды с помощью библиотеки Facebook Prophet,
# Что такое стэкинг и как предсказания моделей использовать как вход для других моделей.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


"""
1 Отбор признаков и поиск гиперпараметров
"""
# датасет ракушек
data = pd.read_csv('abalone.csv')
data = data.rename(columns={"Class_number_of_rings": "rings"})
data = data.rename(columns=str.lower)
data.columns = data.columns.str.replace('_', ' ')

# преобразуем наш категориальный признак
from sklearn.utils import shuffle
# dataset = pd.get_dummies(data.sex).join(data).drop('sex', axis='columns')
# dataset = shuffle(dataset, random_state=1)
# features = list(dataset.columns)
# target = 'rings'
# features.remove(target)

# Отбор признаков
# 1. отбор с помощью взаимной информации между признаками
# 2. отбор с помощью моделей
from sklearn.feature_selection import mutual_info_regression, SelectKBest
# train = int(len(data) * 0.8)
# для классификации есть mutual_info_classif
# regression_best_features = SelectKBest(mutual_info_regression, k=4).fit(
#     dataset[features][:train], dataset[target][:train]
# ).get_support(indices=True)  # этот метод возращает номера колонок
# print("Если кольца считать как регрессию, лучшие 4 признака это", np.array(features)[regression_best_features])

# посчитаем регрессию и классификацию на этих четырех признаках
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
# regressor = MLPRegressor(random_state=1, max_iter=1000).fit(
#     dataset.values[:, regression_best_features][:train],
#     dataset[target][:train])
# можно отбирать признаки в задачах регрессии (и классификации тоже) с помощью взаимной информации
# Интерпретируемость - можем понять, что даёт наибольший вклад в предсказания
# print("R2 test regression %.3f" % (
#     r2_score(dataset[target][train:], regressor.predict(dataset.values[:, regression_best_features][train:]))))

# деревья имеют определенные коэффициенты важности(tree_estimator.feature_importances_) которые показывают,
# какой признак даёт наибольший вклад в предсказание
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
# selector_linear = SelectFromModel(estimator=LinearRegression(normalize=True)).fit(
#     dataset[features][:train], dataset[target][:train])
# selector_forest = SelectFromModel(estimator=RandomForestRegressor(n_estimators=100, random_state=1)).fit(
#     dataset[features][:train], dataset[target][:train])
# linear_features = np.array(features)[selector_linear.get_support()]
# forest_features = np.array(features)[selector_forest.get_support()]
# print("По линейной модели", linear_features)
# print("По случайному лесу", forest_features)

# линейные модели сильно проиграли случайному лесу в отборе признаков на текущих данных
# print("R2 по линейным признакам %.3f (test set)" % (
#     r2_score(
#         dataset[target][train:],
#         MLPRegressor(random_state=1, max_iter=1000).fit(
#             dataset[linear_features][:train], dataset[target][:train]
#         ).predict(
#             dataset[linear_features][train:]))))
# print("R2 по признакам случайного леса %.3f (test set)" % (
#     r2_score(
#         dataset[target][train:],
#         MLPRegressor(random_state=1, max_iter=1000).fit(
#             dataset[forest_features][:train], dataset[target][:train]
#         ).predict(
#             dataset[forest_features][train:]))))

# много вариантов с кросс-валидацией(встроенный в scikit-learn класс GridSearchCV) -
# осуществляет перебор по сетке гиперпараметров
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
# перебор всех сочетаний параметров это может занимать много времени!
# search = GridSearchCV(
#     KNeighborsRegressor(),
#     param_grid={
#         'n_neighbors': [3, 5, 7, 10, 25],
#         'weights': ['uniform', 'distance']
#     }, cv=5
# ).fit(dataset[features], dataset[target])
# print('лучший отобранный по кросс-валидации', search.best_estimator_,
#       'наилучшее значение метрики (R2) %.3f' % search.best_score_)
# отобрали по кросс-валидации лучшие гиперпараметры, перед отправкой модели в среду эксплуатации,
# можем её обучить с найденными параметрами уже на всех данных(в GridSearchCV по умолчанию параметром refit=True)
# если признаков не 5-10 методы могут сильно помочь. Их можно объединять в pipelines(соединять последовательно)


"""
2 Понижение размерности
"""
# Principal component analysis(PCA) - линейное преобразование признаков данных,
# с целью оставить в них как больше информации(как можно больше разброса в величинах)
# PCA работает с разбросом, все величины надо стандартизировать(вычесть среднее и разделить на разброс), чтобы привести к единой шкале
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# отобразим все признаки кроме колец на плоскость
# PCA = make_pipeline(StandardScaler(), PCA(n_components=2))
# transformed = PCA.fit_transform(dataset[features])
# plt.scatter(transformed[:, 0], transformed[:, 1])

# разделение по цвету
# colors = []
# for sample in dataset.values:
#     if sample[0] == 1:
#         colors.append('steelblue')
#     elif sample[1] == 1:
#         colors.append('red')
#     else:
#         colors.append('green')
# plt.scatter(transformed[:, 0], transformed[:, 1], c=colors)

# метод UMAP - укладывает на плоскость все данные(может и в более высокие размерности, но не более размерности данных)
# делает это нелинейно
# запретим предупреждения
import warnings
warnings.filterwarnings('ignore')
from umap import UMAP
# поступим аналогично
# transformed = make_pipeline(
#     StandardScaler(), UMAP()
# ).fit_transform(dataset[features])
# plt.scatter(transformed[:, 0], transformed[:, 1], c=colors)

# разделяются ли данные по полам, если эти признаки пола убрать?
# transformed = make_pipeline(
#     StandardScaler(), UMAP()
# ).fit_transform(dataset[features[3:]])
# пол - это очень важный признак. Если бы мы по остальным признакам различали пол - это плохо получалось
# plt.scatter(transformed[:, 0], transformed[:, 1], c=colors)


"""
3 Кластеризация
"""
# как можно найти кластеры в данных(алгоритм, который разбивает точки на заранее заданное число кластеров):
# 1. На первом шаге инициализируем случайные центроиды кластеров (случайные точки)
# 2. Посчитаем расстояния между каждой точкой и центроидами
# 3. Присвоим номера кластеров точкам по ближайшему расстоянию до конкретного центроида
# 4. Пересчитаем центроиды как центры полученных кластеров
# 5. Повторим шаги 2-4, пока центроиды не перестанут меняться

from sklearn.datasets import load_iris
# заберем только признаки, и только последние два, перемешаем
# X = shuffle(load_iris(return_X_y=True)[0][:, -2:], random_state=1)
# def distance(a, b):
#     return np.sqrt(np.sum((a - b) ** 2))
# def clustering(number_clusters, X):
#     np.random.seed(1)
#     # шаг 1 - случайная инициализация
#     centroids = np.array([
#         [np.random.uniform(X[:, 0].min(), X[:, 0].max()),
#          np.random.uniform(X[:, 1].min(), X[:, 1].max())
#         ] for _ in range(number_clusters)]).reshape(number_clusters, 2)
#     previous = np.zeros_like(centroids)
#     clusters = None
#     iteration = 0
#     while not np.allclose(previous, centroids):
#         previous = centroids.copy()
#         # шаг 2, считаем все расстояния
#         distances = []
#         for centroid in centroids:
#             distances.append([
#                 distance(X[index], centroid) \
#                 for index in range(len(X))])
#         distances = np.array(distances).reshape(number_clusters, len(X))
#         # шаг 3, присваеваем метки точкам
#         clusters = []
#         for index in range(len(X)):
#             clusters.append(
#                 np.argmin(distances[:, index]))
#         clusters = np.array(clusters).reshape(len(X))
#         # шаг 4, пересчитываем центроиды
#         centroids = []
#         for index in range(number_clusters):
#             centroids.append([
#                 X[np.where(clusters == index), 0].mean(),
#                 X[np.where(clusters == index), 1].mean(),])
#         centroids = np.array(centroids).reshape(number_clusters, 2)
#         iteration += 1
#         # print("%d, изменение расстояния центроидов %.4f" % (iteration, distance(previous, centroids)))
#     return centroids, clusters
# centroids, clusters = clustering(3, X)

# центроиды(центры кластеров) на графике
# plt.title("Кластеры и центроиды")
# plt.scatter(X[:, 0], X[:, 1], c=clusters)
# for index, centroid in enumerate(centroids):
#     plt.annotate(
#         "Центроид",
#         (centroid[0], centroid[1]),
#         (centroid[0] + 0.5, centroid[1] - 0.25),
#         arrowprops={"arrowstyle": '->'})

# Для кластеризации существуют метрики качества(одна из интересных - метрик силуэта, silhouette_score)
# Она меняется от -1(худший случай) до +1(лучший), и показывает, насколько “хорошо” принадлежат точка своему кластеру
# (для всех данных - значение усредняется)
# когда мы не знаем количество кластеров заранее - можем перебирать и сравнивать их по метрике силуэта
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# best_score = -1
# best_number = 0
# whole_X = shuffle(load_iris(return_X_y=True)[0], random_state=1)
# переберем от 22 кластеров до 2
# for number_clusters in range(22, 1, -1):
#     clusterer = KMeans(n_clusters=number_clusters, random_state=1).fit(whole_X)
#     score = silhouette_score(whole_X, clusterer.labels_)
#     if score > best_score:
#         best_score = score
#         best_number = number_clusters
        # print("%d: улучшение, метрика силуэта %.3f" % (number_clusters, score))
# plt.title("По всем признакам вот так неочевидно!")
# clusterer = KMeans(n_clusters=best_number, random_state=1).fit(whole_X)
# plt.scatter(X[:, 0], X[:, 1], c=clusterer.labels_)

# методы, которые заранее не требуют задавать количество кластеров(DBSCAN).
# Он требует как минимум задать размер шара, попадая в который соседние для выбранной точки считаются попадающими в одну группу
# plt.figure(figsize=(8, 4))
# plt.title("Все что для выбранной точки попало в шар - считается одной группой")
# plt.gcf().gca().add_artist(plt.Circle(X[len(X) // 2], 0.5, color='red', alpha=0.5))
# plt.scatter(X[:, 0], X[:, 1])
# plt.annotate(
#     "Выбранная точка",
#     X[len(X) // 2],
#     (X[len(X) // 2, 0] - 1, X[len(X) // 2, 1] + 2),
#     arrowprops={"arrowstyle": '->'})
# plt.xlim(0, 8)
# plt.ylim(0, 4)

from sklearn.cluster import DBSCAN
# 1
# dbscan = DBSCAN(eps=0.5).fit(whole_X)
# plt.title("Здесь еще более сложный паттерн\nкластеров: %d, метрика силуэта %.3f" % (
#     len(pd.unique(dbscan.labels_)), silhouette_score(whole_X, dbscan.labels_)))
# plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels_)
# # 2
# dbscan = DBSCAN(eps=1).fit(whole_X)
# plt.title("Здесь радиус вдвое больше\nкластеров: %d, метрика силуэта %.3f" % (
#     len(pd.unique(dbscan.labels_)), silhouette_score(whole_X, dbscan.labels_)))
# plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels_)


"""
4 Поиск аномалий в данных
"""
# пример “аномальные ракушки”.
# первое  - это искать их с помощью деревьев
# IsolationForest использует множество деревьев, каждое по выбранному признаку(с помощью случайных разбиений данных
# при обучении) выдает решение - является пример аномалией(outlier) или нет(inlier)
# считаем наши данные и выберем пару колонок сразу
from sklearn.ensemble import IsolationForest
# isolation = IsolationForest(
#     n_estimators=100,  # количество деревьев
#     contamination=0.1,  # предварительный процент аномалий
# )
# data = data[['length', 'height']]
# print(data.head())
# isolation.fit(data)
# def plot_boundaries_IF(estimator, data, draw_columns):
#     xx, yy = np.meshgrid(
#         np.linspace(data[draw_columns[0]].min() - 0.1, data[draw_columns[0]].max() + 0.1),
#         np.linspace(data[draw_columns[1]].min() - 0.1, data[draw_columns[1]].max() + 0.1))
#     Z = estimator.decision_function(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     plt.title("Где темнее - там скорее аномалии")
#     plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
#     predictions = estimator.predict(data)
#     outliers = predictions < 0
#     inliers = predictions > 0
#     plt.scatter(
#         data[draw_columns[0]].values[inliers],
#         data[draw_columns[1]].values[inliers],
#         c='black', alpha=0.2)
#     plt.scatter(
#         data[draw_columns[0]].values[outliers],
#         data[draw_columns[1]].values[outliers],
#         c='red', alpha=0.8)
#     plt.xlabel(draw_columns[0])
#     plt.ylabel(draw_columns[1])
# plot_boundaries_IF(isolation, data, ['length', 'height'])
# метод Local Outlier Factor(LOF) - в каждой точке оценивает плотность относительно других точек(с помощью близости).
# Чем ниже эта плотность - тем аномальнее пример


"""
5 Одномерные временные ряды
"""
# airpassengers = pd.read_csv('airpassengers.csv', parse_dates=['Month'])
# sns.lineplot(x="Month", y="#Passengers", data=airpassengers)
# import fbprophet
# model = fbprophet.Prophet(
#     daily_seasonality=False, # без внутридневной сезонности
#     weekly_seasonality=False, # и без недельной
#     yearly_seasonality=7 # количество "изгибов" в сезонности
# )
# train = int(len(airpassengers) * 0.8) # длина тренировочных данных
# для fbprophet необходимо передавать колонки с заданными именами
# traininig_data = airpassengers.rename({
#     'Month': 'ds',
#     '#Passengers': 'y'
# }, axis='columns')[:train]
# model.fit(traininig_data)
# future = model.make_future_dataframe(
#     12 + len(airpassengers) - train, # сколько точек наперёд
#     freq='MS' # шагаем по началу месяца (Month Start)
# )
# forecast = model.predict(future)
# forecast[[
#     'ds', 'trend', 'yhat'
# ]].tail()
# plt.title("Аддитивный прогноз с помощью prophet\nкачество на отложенных %d точках R2: %.3f" % (
#     len(airpassengers) - train, r2_score(
#         y_true=airpassengers['#Passengers'].values[train:],
#         y_pred=forecast.yhat.values[train:len(airpassengers)])))
# sns.lineplot(x="Month", y="#Passengers", data=airpassengers, label="Оригинальные данные")
# sns.lineplot(x="ds", y="yhat", data=forecast, label="Прогнозные данные")
# plt.axvline(airpassengers.Month[train], 0, 1, c='black', label='Окончание train')
# plt.legend(loc='best')

# model = fbprophet.Prophet(
#     daily_seasonality=False, # без внутридневной сезонности
#     weekly_seasonality=False, # и без недельной
#     yearly_seasonality=7, # количество "изгибов" в сезонности
#     seasonality_mode='multiplicative')
# train = int(len(airpassengers) * 0.8) # длина тренировочных данных
# для fbprophet необходимо передавать колонки с заданными именами
# traininig_data = airpassengers.rename({
#     'Month': 'ds',
#     '#Passengers': 'y'
# }, axis='columns')[:train]
# model.fit(traininig_data)
# future = model.make_future_dataframe(
#     12 + len(airpassengers) - train, # сколько точек наперёд
#     freq='MS'  # шагаем по началу месяца (Month Start)
# )
# forecast = model.predict(future)
# plt.title("Мультипликативный прогноз с помощью prophet\nкачество на отложенных %d точках R2: %.3f" % (
#     len(airpassengers) - train, r2_score(
#         y_true=airpassengers['#Passengers'].values[train:],
#         y_pred=forecast.yhat.values[train:len(airpassengers)])))
# sns.lineplot(x="Month", y="#Passengers", data=airpassengers, label="Оригинальные данные")
# sns.lineplot(x="ds", y="yhat", data=forecast, label="Прогнозные данные")
# plt.axvline(airpassengers.Month[train], 0, 1, c='black', label='Окончание train')
# plt.legend(loc='best')

# чтобы сделать аддитивную модель мультипликативной,
# достаточно преобразовать целевую величину, например через логарифмирование
from sklearn.preprocessing import MinMaxScaler
# sns.lineplot(
#     x=airpassengers.Month,
#     y=MinMaxScaler().fit_transform(airpassengers["#Passengers"].values.reshape(-1, 1)).flatten(),
#     label="Исходные данные")
# sns.lineplot(
#     x=airpassengers.Month,
#     y=MinMaxScaler().fit_transform(np.log(airpassengers["#Passengers"].values.reshape(-1, 1))).flatten(),
#     label="Логарифмированные данные")
# plt.legend(loc='best')

# fbprophet имеет также полезные функции и для отладки.
# model.plot_components(forecast)
# model.plot(forecast)


"""
6 Стэкинг, или объединение нескольких моделей
"""
# правило стэкинга - обучать вторую модель на тех предсказаниях, которые были получены НЕ с тренировочного множества
# первой модели(нельзя учить на тех же данных, про которые уже известно)
# тренировочных множества должно быть как минимум два
features = list(data.columns)
target = 'rings'
features.remove('sex')
features.remove(target)
from sklearn.model_selection import train_test_split
train_base_X, test_base_X, train_base_y, test_base_y = train_test_split(
    data[features].values, data[target].values.reshape(-1, 1),
    test_size=0.5, random_state=1)
train_stack_X, test_stack_X, train_stack_y, test_stack_y = train_test_split(
    test_base_X, test_base_y,
    test_size=0.5, random_state=1)
# print(train_base_X.shape, train_stack_X.shape, test_stack_X.shape, train_base_y.shape, train_stack_y.shape, test_stack_y.shape)

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
# base_estimator = MLPRegressor(random_state=1).fit(train_base_X, train_base_y)
# stack_estimator = GradientBoostingRegressor(random_state=1).fit(
#     np.column_stack([
#         train_stack_X,
#         base_estimator.predict(train_stack_X)]), train_stack_y)
# "R2 stacked %.3f, R2 base %.3f" % (r2_score(
#     test_stack_y,
#     stack_estimator.predict(
#         np.column_stack([
#             test_stack_X,
#             base_estimator.predict(test_stack_X)]))),
#                                    r2_score(test_stack_y, base_estimator.predict(test_stack_X)))
# стэкинг улучшил результат
# мы уменьшаем количество размеченных данных в случае стэкинга

# классы StackingRegressor и StackingClassifier - обучают последовательность моделей на всех данных, кроме последней,
# для которой они проводят уже обучение на разбиениях
from sklearn.ensemble import StackingRegressor
# regressor = StackingRegressor([
#     ('MLP', MLPRegressor(random_state=1)),
#     ('GBR', GradientBoostingRegressor(random_state=1))
# ]).fit(np.row_stack([
#         train_base_X, train_stack_X]),
#     np.row_stack([train_base_y, train_stack_y]))
# "R2 %.3f" % regressor.score(test_stack_X, test_stack_y)  # С учетом кросс-валидации - практически то же самое

# стэкинг для временных рядов
climate = pd.read_csv('jena_climate_2009_2016.csv', parse_dates=['Date Time'], dayfirst=True)
# выберем за целевую величину температуру в градусах,
# за признаки - давление, влажность в процентах, скорость ветра(м/с) и направление в градусах
features = ['p (mbar)', 'rh (%)', 'wv (m/s)', 'wd (deg)', 'Date Time']
target = 'T (degC)'
# sns.lineplot(x="Date Time", y=target, data=climate[:20])

# не интересуют все 420тыс записей(каждые 10 минут), только дневные, поэтому переиндексировать таблицу в среднедневные значения
dataset = climate[features + [target]].set_index('Date Time').resample('D').mean()
# print(dataset.describe())
# sns.pairplot(data=dataset[:365])

# проверка - связаны ли признаки
# удалить пропуски
data = dataset.dropna()
# sns.lineplot(data=(data[:365] - data.min()) / (data.max() - data.min()))

# Разгладить значения скользящим средним
# plt.plot(data['T (degC)'][:365], label='Original')
# plt.plot(data['T (degC)'].rolling(window=20, center=True).mean()[:365], label='Filtered')
# plt.legend(loc='best')
# потеря некоторой информации о погодных аномалиях(всреднем не сильно большие отличия)

# применим ко всем данным и стандартизируем данные
data = data.rolling(window=20, center=True).mean().dropna()
mean = data.mean()
std = data.std()
data = (data - mean) / std

def get_train_test_features(train_size, forecast_steps):
    result = {}
    features_without_date = list(data.columns)
    features_without_date.remove('T (degC)')
    for feature in features_without_date:
        model = fbprophet.Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=365 // 10
        ).fit(
            data[[feature]].reset_index().rename({
                'Date Time': 'ds',
                feature: 'y'
            }, axis='columns')[:train_size])
        future = model.make_future_dataframe(forecast_steps, include_history=False)
        result[feature] = model.predict(future).yhat.values
    return result
features_train = 365 * 3 + 180
future_steps = len(data) - features_train + 365
features_forecast = get_train_test_features(
    train_size=features_train,
    forecast_steps=future_steps)
features_data = pd.DataFrame(
    features_forecast,
    index=pd.date_range(start=data.index[features_train], periods=future_steps))
train = int(future_steps * 0.5)
test = len(data) - features_train - train

# поскольку у нас здесь обычная регрессия, мы можем примеры перемешать
X, y = shuffle(
    features_data[:train],
    data[target].values[features_train:features_train + train],
    random_state=1)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer
target_regressor = make_pipeline(
    PolynomialFeatures(degree=2),
    PowerTransformer(),
    LinearRegression()).fit(X, y)
"R2 test (%d samples) %.3f" % (test, target_regressor.score(
    features_data[train:train + test], data[target].values[
                                       features_train + train:features_train + train + test]))
from sklearn.metrics import mean_absolute_error
t_original = dataset[target].dropna().values[
    features_train + train:features_train + train + test]
t_predicted = target_regressor.predict(features_data[train:]) * std[target] + mean[target]
plt.figure(figsize=(14, 7))
plt.title('Как выглядит предсказание, средняя ошибка в градусах %.1f' % mean_absolute_error(
    t_original,
    t_predicted[:test]))
plt.plot(
    features_data[train:train + test].index,
    data[target].values[
        features_train + train:features_train + train + test
    ] * std[target] + mean[target],
    label="Скользящая 31-дневная средняя TRUE DATA",
    lw=2)
plt.plot(
    features_data[train:train + test + 365].index,
    t_predicted[:test + 365],
    label='PRED DATA',
    lw=2)
plt.plot(
    features_data[train:train + test].index,
    t_original,
    label='TRUE DATA',
    alpha=0.25)
plt.legend(loc='best')
# средняя точность до 4 градусов