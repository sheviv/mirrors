# https://mlbootcamp.ru/ru/article/tutorial/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')



"""
Загружаем данные
"""
# задача определения кредитной платежеспособности(кредитного скрининга)
# header = None - таблица не содержит заголовка(не содержит названий столбцов)
# na_values = '?' - данные содержат пропущенные значения, обозначенные символом ?
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'
data = pd.read_csv(url, header=None, na_values='?')



"""
Анализируем данные
"""
# 690 строк (объектов) и 16 столбцов (признаков)
# print(data.shape)
# все значения категориальных признаков заменены символами, числовые признаки приведены к другому масштабу
# Последний столбец содержит символы + и -(вернул кредит/нет)
data.columns = ['A' + str(i) for i in range(1, 16)] + ['class']  # зададим столбцам имена
# print(data.head())
# при обращении в квадратных сокбках указывается имя столбца, затем – строки
a = data.at[687, 'A5']
# describe() получим некоторую сводную информацию по всей таблице(только для количественных признаков)
# количество(count) среднее(mean) стандартное отклонение(std) медиана(50%) нижний(25%) верхний(75%) квартиль
aa = data.describe()
# количество элементов в столбцах A2, A14 меньше общего количества объектов (690)(столбцы содержат пропущенные значения)
# Выделим числовые и категориальные признаки
categorical_columns = [c for c in data.columns if data[c].dtype.name == 'object']
numerical_columns = [c for c in data.columns if data[c].dtype.name != 'object']
# общуая информацию по категориальным и числовым признакам
# число заполненных(count) значения принимаемые признаком(unique) самое популярное(top) количество объектов где самое частое значение признака(freq)
data[categorical_columns].describe()
data[numerical_columns].describe()
# или
data.describe(include=[object])
# перечень значений категориальных признаков
# for c in categorical_columns:
    # print(data[c].unique())
# scatter_matrix(pandas.tools.plotting) - для каждой количественной переменной гистограмма, для каждой пары таких переменных – диаграмму рассеяния
from pandas.plotting import scatter_matrix
# sc = scatter_matrix(data, alpha=0.05, figsize=(10, 10))
# plt.show()
# корреляционная матрица
data.corr()
# диаграмму рассеяния для пары признаков(разные классы: + – красный, - – синий)(A2, A11)
col1 = 'A2'
col2 = 'A11'
plt.figure(figsize=(10, 6))
plt.scatter(data[col1][data['class'] == '+'],
            data[col2][data['class'] == '+'],
            alpha=0.75,
            color='red',
            label='+')
plt.scatter(data[col1][data['class'] == '-'],
            data[col2][data['class'] == '-'],
            alpha=0.75,
            color='blue',
            label='-')
plt.xlabel(col1)
plt.ylabel(col2)
plt.legend(loc='best')
plt.show()
# признак A11 - существенный:красные точки имеют большое значение этого признака, синие – маленькое.
# визуально наблюдается хорошая корреляция между признаками A11 и class.
# A2 несет гораздо меньше информации о принадлежности объекта интересующему нас классу.



"""
Готовим данные
"""
# scikit-learn не работают напрямую с категориальными признаками и данными, в которых имеются пропущенные значения
# Пропущенные значения. заполненные(непропущенных) элементы - count.
# axis = 0 - мы двигаемся по 0(сверху вниз), а не размерности 1(слева направо)
data.count(axis=0)
# Если данные содержат пропущенные значения, то:
data = data.dropna(axis=1)  # удалить столбцы с такими значениями
data = data.dropna(axis=0)  # удалить строки с такими значениями
# данных может стать совсем мало, поэтому:

# Количественные признаки
# Заполнить пропущенные значения можно с помощью метода fillna(Заполним, например, медианными значениями)
data = data.fillna(data.median(axis=0), axis=0)  # axis=0 - двигаемся сверху вниз
# Проверим, что теперь все столбцы, соответствующие количественным признакам, заполнены
data.count(axis=0)

# Категориальные признаки
# заполнение пропущенных значений самым популярным в столбце. Начнем с A1
data['A1'].describe()
# Заполняем все пропуски b значением(встречается 468 - чаще всех)
data['A1'] = data['A1'].fillna('b')
# Автоматизируем процесс:
data_describe = data.describe(include=[object])
for c in categorical_columns:
    data[c] = data[c].fillna(data_describe[c]['top'])
# Теперь все элементы таблицы заполнены:
data.describe(include=[object])



"""
Векторизация
"""
# преобразуем категориальные признаки в количественные
# бинарные признаки и принимающие большее количество значений будем обрабатываются по-разному
# выделим бинарные и небинарные признаки
binary_columns = [c for c in categorical_columns if data_describe[c]['unique'] == 2]
nonbinary_columns = [c for c in categorical_columns if data_describe[c]['unique'] > 2]
# Бинарные признаки
# Значения бинарных признаков заменим на 0 и 1. Начнем с A1
data.at[data['A1'] == 'b', 'A1'] = 0
data.at[data['A1'] == 'a', 'A1'] = 1
data['A1'].describe()
data_describe = data.describe(include=[object])
# Автоматизируем процесс:
for c in binary_columns[1:]:
    top = data_describe[c]['top']
    top_items = data[c] == top
    data.loc[top_items, c] = 0
    data.loc[np.logical_not(top_items), c] = 1
# Итог
data[binary_columns].describe()

# Небинарные признаки
# К небинарными признакам применим метод векторизации.
data['A4'].unique()  # array(['u', 'y', 'l'], dtype=object)
# Заменим признак A4 тремя признаками: A4_u, A4_y, A4_l.
# Если признак A4 принимает значение u, то признак A4_u равен 1, A4_y равен 0, A4_l равен 0.
# Если признак A4 принимает значение y, то признак A4_y равен 0, A4_y равен 1, A4_l равен 0.
# Если признак A4 принимает значение l, то признак A4_l равен 0, A4_y равен 0, A4_l равен 1.
data_nonbinary = pd.get_dummies(data[nonbinary_columns])
cl = data_nonbinary.columns



"""
Нормализация количественных признаков
"""
# алгоритмы машинного обучения чувствительны к масштабированию данных. количественные признаки полезно нормализовать.
# Например, каждый количественный признак приведем к нулевому среднему и единичному среднеквадратичному отклонению
data_numerical = data[numerical_columns]
data_numerical = (data_numerical - data_numerical.mean()) / data_numerical.std()



"""
Соединяем все в одну таблицу
"""
data = pd.concat((data_numerical, data[binary_columns], data_nonbinary), axis=1)
data = pd.DataFrame(data, dtype=float)
# отдельно рассмотрим столбцы, соответствующие входным признакам(матрица X), а отдельно – выделенный признак(вектор y)
X = data.drop(('class'), axis=1)  # Выбрасываем столбец 'class'.
y = data['class']
feature_names = X.columns
N, d = X.shape  # Теперь 42 входных признака



"""
Обучающая и тестовая выборки
"""
# Обычно используют разбиения в пропорции 50%:50%, 60%:40%, 75%:25%
# разобьем данные на обучающую/тестовую выборки в отношении 70%:30%:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)  # X_train, y_train – это обучающая, X_test, y_test – тестовая.
N_train, _ = X_train.shape
N_test,  _ = X_test.shape



"""
Алгоритмы машинного обучения
"""
# Метод	                                     Класс
# kNN – k ближайших соседей	                 sklearn.neighbors.KNeighborsClassifier
# LDA – линейный дискриминантный анализ	     sklearn.lda.LDA
# QDA – квадратичный дискриминантный анализ  sklearn.qda.QDA
# Logistic – логистическая регрессия	     sklearn.linear_model.LogisticRegression
# SVC – машина опорных векторов	             sklearn.svm.SVC
# Tree – деревья решений	                 sklearn.tree.DecisionTreeClassifier
# RF – случайный лес	                     sklearn.ensemble.RandomForestClassifier
# AdaBoost – адаптивный бустинг	             sklearn.ensemble.AdaBoostClassifier
# GBT – градиентный бустинг деревьев решений sklearn.ensemble.GradientBoostingClassifier

# Основные методы классов, реализующих алгоритмы машинного обучения
# Метод класса	        Описание
# fit(X, y)	            обучение (тренировка) модели на обучающей выборке X, y
# predict(X)	        предсказание на данных X
# set_params(**params)	установка параметров алгоритма
# get_params()	        чтение параметров алгоритма


"""
kNN – метод ближайших соседей
"""
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
# качество построенной модели
y_train_predict = knn.predict(X_train)
y_test_predict = knn.predict(X_test)
# err_train и err_test – это ошибки на обучающей и тестовой выборках
err_train = np.mean(y_train != y_train_predict)
err_test = np.mean(y_test != y_test_predict)
# Поиск оптимальных значений параметров с помощью класса GridSearchCV – поиск наилучшего набора параметров,
# доставляющих минимум ошибке перекрестного контроля (cross-validation)
from sklearn.model_selection import GridSearchCV
n_neighbors_array = [1, 3, 5, 7, 10, 15]
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid={'n_neighbors': n_neighbors_array})
grid.fit(X_train, y_train)
best_cv_err = 1 - grid.best_score_
best_n_neighbors = grid.best_estimator_.n_neighbors
# Проверим, чему равны ошибки на обучающей и тестовой выборках при этом значении параметра
knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)
knn.fit(X_train, y_train)
err_train = np.mean(y_train != knn.predict(X_train))
err_test = np.mean(y_test != knn.predict(X_test))


"""
SVC – машина опорных векторов
"""
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
err_train = np.mean(y_train != svc.predict(X_train))
err_test = np.mean(y_test != svc.predict(X_test))

# Радиальное ядро
# попробуем найти лучшие значения параметров для радиального ядра
from sklearn.model_selection import GridSearchCV
C_array = np.logspace(-3, 3, num=7)
gamma_array = np.logspace(-5, 2, num=8)
svc = SVC(kernel='rbf')
grid = GridSearchCV(svc, param_grid={'C': C_array, 'gamma': gamma_array})
grid.fit(X_train, y_train)
# Посмотрим, чему равна ошибка на тестовой выборке при найденных значениях параметров алгоритма:
svc = SVC(kernel='rbf', C=grid.best_estimator_.C, gamma=grid.best_estimator_.gamma)
svc.fit(X_train, y_train)
err_train = np.mean(y_train != svc.predict(X_train))
err_test = np.mean(y_test != svc.predict(X_test))

# Линейное ядро
from sklearn.model_selection import GridSearchCV
C_array = np.logspace(-3, 3, num=7)
svc = SVC(kernel='linear')
grid = GridSearchCV(svc, param_grid={'C': C_array})
grid.fit(X_train, y_train)
svc = SVC(kernel='linear', C=grid.best_estimator_.C)
svc.fit(X_train, y_train)
err_train = np.mean(y_train != svc.predict(X_train))
err_test = np.mean(y_test != svc.predict(X_test))

# Полиномиальное ядро
from sklearn.model_selection import GridSearchCV
C_array = np.logspace(-5, 2, num=8)
gamma_array = np.logspace(-5, 2, num=8)
degree_array = [2, 3, 4]
svc = SVC(kernel='poly')
grid = GridSearchCV(svc, param_grid={'C': C_array, 'gamma': gamma_array, 'degree': degree_array})
grid.fit(X_train, y_train)
svc = SVC(kernel='poly', C=grid.best_estimator_.C,
          gamma=grid.best_estimator_.gamma, degree=grid.best_estimator_.degree)
svc.fit(X_train, y_train)
err_train = np.mean(y_train != svc.predict(X_train))
err_test = np.mean(y_test != svc.predict(X_test))



"""
Random Forest – случайный лес
"""
# ансамбли случайных деревьев, каждое обучается на выборке, полученной из исходной с помощью процедуры изъятия с возвращением
from sklearn import ensemble
rf = ensemble.RandomForestClassifier(n_estimators=100, random_state=11)
rf.fit(X_train, y_train)
err_train = np.mean(y_train != rf.predict(X_train))
err_test = np.mean(y_test != rf.predict(X_test))
# Упорядочим значимости признаков и выведем их значения:
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
# print("Feature importances:")
for f, idx in enumerate(indices):
    pass
    # print("{:2d}. feature '{:5s}' ({:.4f})".format(f + 1, feature_names[idx], importances[idx]))
# Построим столбцовую диаграмму, графически представляющую значимость первых 20 признаков:
d_first = 20
plt.figure(figsize=(8, 8))
plt.title("Feature importances")
plt.bar(range(d_first), importances[indices[:d_first]], align='center')
plt.xticks(range(d_first), np.array(feature_names)[indices[:d_first]], rotation=90)
plt.xlim([-1, d_first])
best_features = indices[:8]
best_features_names = feature_names[best_features]   # основную роль играют признаки A9, A8, A11, A15, A3, A14, A2, A10



"""
GBT – градиентный бустинг деревьев решений
"""
# На каждой итерации строится новый классификатор, аппроксимирующий значение градиента функции потерь
from sklearn import ensemble
gbt = ensemble.GradientBoostingClassifier(n_estimators=100, random_state=11)
gbt.fit(X_train, y_train)
err_train = np.mean(y_train != gbt.predict(X_train))
err_test = np.mean(y_test != gbt.predict(X_test))
gbt = ensemble.GradientBoostingClassifier(n_estimators=100, random_state=11)
gbt.fit(X_train[best_features_names], y_train)
err_train = np.mean(y_train != gbt.predict(X_train[best_features_names]))
err_test = np.mean(y_test != gbt.predict(X_test[best_features_names]))
