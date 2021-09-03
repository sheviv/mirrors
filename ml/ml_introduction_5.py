# https://timeseries-ru.github.io/course/day_5.html
# Курс “Введение в анализ данных и машинное обучение”

# некоторые методы локальной интерпретации
# как создать веб-сервис на flask, как сохранить и загрузить модель
# презентация jupyter notebooks на voila и немного streamlit
# элементы мониторинга после внедрения
# немного о вероятностном программировании
# вариационные автокодировщики (будем генерировать картинки).

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')


"""
1 Методы интерпретации
"""
# методы интерпретации модели могут быть глобальными(как важность признаков у деревьев), а могут быть локальными
# shap, которая такие предсказания и позволяет делать
# загрузим датасет ракушек
data = pd.read_csv('abalone.csv')
data = data.rename(columns={"Class_number_of_rings": "rings"})
data = data.rename(columns=str.lower)
data.columns = data.columns.str.replace('_', ' ')
# немного препроцессинга
# data['sex'] = data.sex.apply(lambda sex: {'F': 0, 'M': 1, 'I': 2}[sex])
# print(data.sample(3))

# shap может объяснять модели на базе деревьев(TreeExplainer), линейные (LinearExplainer, требуются независимые(несвязанные) - признаки),
# нейронные сети keras(DeepExplainer), произвольные модели(KernelExplainer) - весьма медленно
import shap

# результаты работы регрессора
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import shuffle
# отложим 100 сэмплов
# indices = shuffle(list(range(len(data))), random_state=1)
# train = indices[:-100]
# test = indices[-100:]
# regressor = GradientBoostingRegressor(
#     n_estimators=15, max_depth=7, random_state=1
# ).fit(data.iloc[train,:-1], data.iloc[train, -1])
# print("test score %.3f" % regressor.score(data.iloc[test, :-1], data.iloc[test, -1]))
# explainer = shap.TreeExplainer(regressor, data=data.iloc[test, :-1])
# shap_values_test = explainer.shap_values(data.iloc[test, :-1])
# print('True rings value for first test sample = %d' % data.iloc[test[0], -1])
# красным - то что добавило нам колец, а синим - то, что(для этой ракушки) убавило
# shap.force_plot(explainer.expected_value, shap_values_test[0], data.iloc[test[0], :-1], matplotlib=True)

# from sklearn.ensemble import GradientBoostingClassifier
# classifier = GradientBoostingClassifier(n_estimators=15, max_depth=3, random_state=1
# ).fit(data.iloc[train,1:], data.iloc[train, 0])
# print("test score %.3f" % classifier.score(data.iloc[test, 1:], data.iloc[test, 0]))

from alibi.confidence import TrustScore
# ts = TrustScore(
#     # тут могут быть настройки, но мы
#     # оставим всё по умолчанию
# )

# ts.fit(
#     data.iloc[train,1:].values,
#     data.iloc[train, 0].values,
#     classes=3)
# predicted = classifier.predict([data.iloc[test[0], 1:]])[0]
# print("True class %d, predicted %d" % (
#     data.iloc[test[0], 0],
#     predicted))
#
# trust_scores, closest_classes = ts.score(
#     np.array([data.iloc[test[0], 1:]]),
#     np.array([predicted]))
# print("Proba %.3f, TrustScore %.3f, closest class %d" % (
#     classifier.predict_proba(
#         data.iloc[test[0], 1:].values.reshape(1, -1)
#     )[0][0],
#     trust_scores[0], closest_classes[0]))  # trust score < 1 , и ближайший класс точнее

# техника позволяющая одновременно отбирать признаки и строить некоторую интерпретацию вида “мы тут можем ошибаться на столько-то”
# indices = shuffle(list(range(len(data))), random_state=1)
# train = indices[:-2000]
# val = indices[-2000:-1000]
# test = indices[-1000:]
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer
from sklearn.pipeline import make_pipeline
# preprocessor_regression = make_pipeline(PowerTransformer(), PolynomialFeatures(degree=3))
# X_regression_train = preprocessor_regression.fit_transform(data.iloc[train,1:-1])
# X_regression_val = preprocessor_regression.transform(data.iloc[val,1:-1])
# X_regression_test = preprocessor_regression.transform(data.iloc[test,1:-1])
# preprocessor_classification = make_pipeline(PowerTransformer(), PolynomialFeatures(degree=3))
# X_classification_train = preprocessor_classification.fit_transform(data.iloc[train,1:])
# X_classification_val = preprocessor_classification.transform(data.iloc[val,1:])
# X_classification_test = preprocessor_classification.transform(data.iloc[test,1:])
# regressor = Ridge(random_state=1).fit(
#     X_regression_train, data.iloc[train, -1])
# classifier = LogisticRegression(random_state=1).fit(
#     X_classification_train, data.iloc[train, 0])
# print("simple: r2 %.3f, acc %.3f" % (
#     regressor.score(X_regression_test, data.iloc[test, -1]),
#     classifier.score(X_classification_test, data.iloc[test, 0])))

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
# def get_error_models(regression_features, classifier_features):
#     error_regressor = GradientBoostingRegressor(
#         n_estimators=30, max_depth=5, random_state=1
#     ).fit(
#         X_regression_val[:, regression_features],
#         np.abs(
#             regressor.predict(X_regression_val[:, regression_features]) - data.iloc[val, -1]))
#     error_classifier = GradientBoostingClassifier(
#         n_estimators=30, max_depth=5, random_state=1
#     ).fit(
#         X_classification_val[:, classifier_features],
#         (classifier.predict(X_classification_val[:, classifier_features]) != data.iloc[val, 0]
#         ).astype(int))
#     return error_regressor, error_classifier
# error_regressor, error_classifier = get_error_models(
#     regression_features=slice(None),
#     classifier_features=slice(None))

# получен регрессор ошибки и классификатор ошибки. Их можно использовать и для отбора признаков
# (в обоих случаях уберем несколько неважных признаков и посмотрим)
# redundand = 10
# regressor_features = np.argsort(error_regressor.feature_importances_)[redundand:]
# classifier_features = np.argsort(error_classifier.feature_importances_)[redundand:]
# regressor = Ridge(random_state=1).fit(
#     X_regression_train[:, regressor_features],
#     data.iloc[train, -1])
# classifier = LogisticRegression(random_state=1).fit(
#     X_classification_train[:, classifier_features],
#     data.iloc[train, :1])
# print("selected: r2 %.3f, acc %.3f" % (
#     regressor.score(X_regression_test[:, regressor_features], data.iloc[test, -1]),
#     classifier.score(X_classification_test[:, classifier_features], data.iloc[test, 0])))
# убраны признаки, которые давали наименьший вклад в объяснение ошибки - они имеют слабое влияния на предсказания изначальной модели

# ошибка предсказания для конкретного примера
# error_regressor, error_classifier = get_error_models(
#     regressor_features, classifier_features)
# print("regression: True %d, Predicted %.2f, error %.3f" % (
#     data.iloc[test[0], -1],
#     regressor.predict([X_regression_test[0, regressor_features]])[0],
#     error_regressor.predict([X_regression_test[0, regressor_features]])[0]))
# print("classification: True %s, Predicted %s, is error %s" % (
#     str(data.iloc[test[0], 0].astype(int)),
#     str(classifier.predict([X_classification_test[0, classifier_features]])[0]),
#     str(error_classifier.predict([X_classification_test[0, classifier_features]])[0])))


"""
2 Развертывание моделей
"""
# Модели scikit-learn могут быть сериализованы встроенными средствами pickle. Для keras методы save/load
# model = keras.Model()
# filepath = 'path/to/filename.h5'
# ...
# метод save
# model.save(filepath)
# model = keras.models.load_model(filepath)

# датасет нарисованных чисел
from sklearn.datasets import load_digits
# X, y = shuffle(*load_digits(return_X_y=True), random_state=1)
# plt.title("This is number %d" % y[0])
# plt.imshow(X[0].reshape(8, 8), cmap='gray')
# plt.show()

from sklearn.neural_network import MLPClassifier
# def preprocess_samples(samples):
#     return samples / X.max()
# train = int(len(y) * 0.8)
# digitizer = MLPClassifier([32, 16], random_state=1).fit(
#     preprocess_samples(X[:train]), y[:train])
# "Accuracy score %.3f" % digitizer.score(preprocess_samples(X[train:]), y[train:])

# теперь сохраним всё в файл
import pickle
# with open('models/digitizer.pickle', 'wb') as fd:
#     # wb - write binary
#     pickle.dump({
#         'scale': X.max(),
#         'model': digitizer
#     }, file=fd)

# Сервер выглядит вот так (код можно найти в code/digitizer_service.py)
# from flask import Flask, request, jsonify
# app = Flask(__name__)
import pickle
# with open('../models/digitizer.pickle', 'rb') as fd:
#     methods = pickle.load(fd)
# scale = methods['scale']
# model = methods['model']
from json import loads
# @app.route('/', methods=['POST'])
# def index():
#     data = loads(request.json)
#     result = {'prediction': int(model.predict(np.array([data]).astype(float) / scale)[0])}
#     return jsonify(result)
# файл запустили напрямую
# if __name__ == '__main__':
#     app.run(host='localhost', port=5555, debug=True)

# Его необходимо запустить отдельно (например через python3 digitizer_service.py), чтобы протестировать.
import json, requests
# response = requests.post('http://localhost:5555/', json=json.dumps(X[-1].tolist()))
# json.loads(response.content), {'true label': y[-1]}


"""
3 Презентация моделей
"""
# два метода - voila и streamlit:
# 1. voila - это пакет, позволяющий одной командой запустить jupyter notebook (без отображения кода по умолчанию) как интерактивную веб-страницу,
# 2. streamlit - это пакет, позволяющий без помощи jupyter создавать интерактивные веб-страницы с помощью скриптов прямо на python.


"""
4 A/B-тестирование
"""
# эффект от модели = эффект с моделью - эффект без модели
# control_size = 5000
# control_errors = 1000
# test_size = 2000
# test_errors = 380
# print("error rate control %.2f, test %.2f" % (
#     control_errors / control_size, test_errors / test_size))

# Beta-распределение - это такое распределение, что после применения теоремы Байеса мы получим то же распределение с другими параметрами
# def check_distribution(errors, correct):
#     np.random.seed(1)
#     # случайные вероятности "успеха", пусть 100 миллионов
#     percents = np.random.uniform(size=10 ** 8)
#     # биномиальное распределение, длина серии = ошибки + корректные
#     trials = np.random.binomial(n=errors + correct, p=percents)
#     # посмотрим для сравнения сразу Beta-распределение
#     beta_samples = np.random.beta(errors + 1, correct + 1, size=10 ** 5)
#     sns.distplot(beta_samples, hist=False, label="Beta", color='steelblue')
#     # выберем проценты, которые дают то же количество ошибок из общего числа
#     samples = percents[
#         np.where(trials == errors)]
#     # и отобразим распределение этих процентов
#     sns.distplot(samples, hist=False, label="Sampled", color='red')
#     plt.axvline(beta_samples.mean(), 0, 1, ls='--', c='black')
#     plt.axvline(samples.mean(), 0, 1, ls='--', c='black')
#     plt.legend(loc='best')
# запустим по 100 тысяч успехов и неуспехов
# check_distribution(10 ** 5, 10 ** 5)


# def plot_beta(errors, correct, label, samples=None):
#     np.random.seed(1)
#     X = np.random.beta(errors + 1, correct + 1, size=(errors + correct) if samples is None else samples)
#     sns.distplot(X, hist=False, label=label)
#     plt.axvline(X.mean(), 0, 1, ls='--', c='black', alpha=0.25)
#     plt.xlim(0.15, 0.25)
#     plt.xlabel('Вероятность ошибки')
#     plt.ylabel('Оценка плотности')
# plt.title("После первых пошедших данных")
# plot_beta(control_errors, control_size - control_errors, 'Control')
# plot_beta(test_errors, test_size - test_errors, 'Test')

# несмотря на схожесть средних процентов, уверенности в том что вариант с моделью лучше - у нас не так уж и много
# rescale = 10
# new_control_errors = control_errors * rescale
# new_control_size = control_size * rescale
# new_test_errors = test_errors * rescale
# new_test_size = test_size * rescale
# plt.title("Увеличили количество данных в %d раз" % rescale)
# plot_beta(control_errors + new_control_errors, control_size + new_control_size - control_errors - new_control_errors, 'Control')
# plot_beta(test_errors + new_test_errors, test_size + new_test_size - test_errors - new_test_errors, 'Test')
# тестовый вариант с моделью лучше


"""
5 Вероятностное программирование
"""
# локальные хаки, могут быть не нужны
import os
import pymc3
# os.environ['MKL_THREADING_LAYER'] = 'GNU'
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle

# male = data[data.sex == 'M'].drop('sex', axis='columns')
# возьмемся предсказывать количество колец
# target = 'rings'
# train = male[:2 * len(male) // 3]
# test = male[2 * len(male) // 3:]

# Параметры линейной зависимости не известны, вывести их распределение из данных путем сэмплирования
# def analyze_data(data):
#     model = pymc3.Model()
#     with model:
#         # распределения коэффициентов линейной модели
#         # mu - среднее, sd - разброс. Это всё приоры
#         alpha = pymc3.Normal('alpha', mu=train[target].mean() / 2 / train['diameter'].mean(), sd=1)
#         beta = pymc3.Normal('beta', mu=train[target].mean() / 2 / train['whole weight'].mean(), sd=1)
#         # свободное слагаемое в линейной модели будет с некоторым неизвестным разбросом
#         # мы получаем иерархическую модель. Экспоненциальное распределение - со средним 1 / lambda,
#         # моделирует неотрицательную вещественную величину с убывающей вероятностью больших значений
#         sd = pymc3.Exponential('deviation', lam=1 / 3)
#         gamma = pymc3.Normal('gamma', mu=0, sd=sd)
#         estimate = alpha * train['diameter'] + beta * train['whole weight'] + gamma
#         observed = pymc3.Poisson('rings', mu=estimate, observed=train[target])
#         # MAP - maximum aposteriori - это попытка найти хорошую стартовую точку для начала сэмплирования
#         MAP = pymc3.find_MAP()
#         # tune - число "разогревочных" сэмплов, мы их рассматривать не будем
#         traces = pymc3.sample(draws=500, tune=500, start=MAP, discard_tuned_samples=True, cores=2, chains=2)
#     # поскольку у нас 2 цепочки по 500, в каждом следе мы возьмем каждый второй сэмпл, чтобы снизить влияние зависимости следующего от предыдущего
#     predictions = traces['alpha'][::4].mean() * test['diameter'] + traces['beta'][::4].mean() * test['whole weight'] + \
#                   traces['gamma'][::4].mean()
#     return model, traces, mean_absolute_error(test[target], predictions)
# male_model, male_traces, male_mae = analyze_data(male)
# print("TEST MAE: %.1f" % male_mae)

# множество значений параметров
# Чем более случайно перемешаны параметры - тем лучше
# with male_model:
#     pymc3.plot_trace(male_traces, var_names=['alpha', 'beta', 'gamma'])

# определяются некоторые границы для большинства ракушек путем операций над сэмплированными параметрами
# def male_predict(data, operation=np.mean, randomize=False):
#     np.random.seed(1)
#     alpha = operation(male_traces['alpha'][::4])
#     beta = operation(male_traces['beta'][::4])
#     gamma = operation(male_traces['gamma'][::4])
#     mean = alpha * data['diameter'] + beta * data['whole weight'] + gamma
#     if randomize:
#         return np.random.poisson(lam=mean)
#     return mean
# plt.scatter(test.diameter, test[target], alpha=0.2, label='Original')
# plt.scatter(test.diameter, male_predict(test), alpha=0.2, label='Mean Prediction')
# plt.scatter(test.diameter, male_predict(test, operation=np.max), alpha=0.2, label='Max Prediction')
# plt.scatter(test.diameter, male_predict(test, operation=np.min), alpha=0.2, label='Min Prediction')
# plt.xlabel('diameter')
# plt.ylabel('rings')
# plt.legend(loc='best')
# # или
# plt.scatter(test['whole weight'], test[target], alpha=0.25, label='Original')
# plt.scatter(test['whole weight'], male_predict(test), alpha=0.25, label='Mean Prediction')
# plt.scatter(test['whole weight'], male_predict(test, operation=np.max), alpha=0.25, label='Max Prediction')
# plt.scatter(test['whole weight'], male_predict(test, operation=np.min), alpha=0.25, label='Min Prediction')
# plt.xlabel('whole weight')
# plt.ylabel('rings')
# plt.legend(loc='best')

# целевая величина случайная, можем получать её (случайные) значения просто выбирая случайные значения параметров
# plt.scatter(test['diameter'], test[target], label='Original')
# for _ in range(10):
#     plt.scatter(test['diameter'], male_predict(test, operation=np.random.choice, randomize=True), c='red', alpha=0.05)
# plt.xlabel('diameter')
# plt.ylabel('rings')


"""
6 Генеративные задачи
"""
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # -1 для СPU
import tensorflow
from tensorflow import keras
