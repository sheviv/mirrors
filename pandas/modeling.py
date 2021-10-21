# Переход от переформатирования данных к подгонке и оцениванию моделей.

import pandas as pd
import numpy as np

# преобразования объекта DataFrame в массив NumPy
data = pd.DataFrame({'x0': [1, 2, 3, 4, 5],
                     'x1': [0.01, -0.01, 0.25, -4.1, 0.],
                     'y': [-1.5, 0., 3.6, 1.3, -2.]})
# print(data.columns)
# print(data.values)
# ///
# обратное преобразования в DataFrame(передать двумерный массив ndarray и имена столбцов)
# df2 = pd.DataFrame(data.values, columns=['one', 'two', 'three'])


# Описание моделей с помощью Patsy
# описание статистических моделей
import patsy
y, X = patsy.dmatrices('y ~ x0 + x1', data)
# Свободный член + 0
cv = patsy.dmatrices('y ~ x0 + x1 + 0', data)[1]
# print(cv)
# Patsy передать напрямую в алгоритм
coef, resid, _, _ = np.linalg.lstsq(X, y)

# Преобразование данных в формулах Patsy
# y, X = patsy.dmatrices('y ~ x0 + np.log(np.abs(x1) + 1)', data)

# ///
# стандартизацию (приведение к распределению со средним 0 и дисперсией 1) и центрирование (вычитание среднего)
# y, X = patsy.dmatrices('y ~ standardize(x0) + center(x1)', data)
# ///
# сложить именованные столбцы из набора данных(обернуть операцию специальной функцией I)
# y, X = patsy.dmatrices('y ~ I(x0 + x1)', data)


# Statsmodels – для подгонки разнообразных статистических моделей, выполнения статистических тестов,
# исследования и визуализации данных.
# Оценивание линейных моделей(Линейные модели в statsmodels - два основных интерфейса:
# на основе массивов
# на основе формул)
import statsmodels.api as sm
import statsmodels.formula.api as smf
# сгенерировать линейную модель по случайным данным
def dnorm(mean, variance, size=1):
    if isinstance(size, int):
        size = size
    return mean + np.sqrt(variance) * np.random.randn(*size)
# Для вопроизводимости результатов
np.random.seed(12345)
N = 100
X = np.c_[dnorm(0, 0.4, size=N),
          dnorm(0, 0.6, size=N),
          dnorm(0, 0.2, size=N)]
eps = dnorm(0, 0.1, size=N)  # генерации нормально распределенных данных с заданными средним и дисперсией.
beta = [0.1, 0.3, 0.5]
y = np.dot(X, beta) + eps
# добавить столбец свободного члена в существующую матрицу
X_model = sm.add_constant(X)
# sm.OLS - линейная регрессия(методом наименьших квадратов)
model = sm.OLS(y, X)
# fit возвращает объект с результатами регрессии
results = model.fit()
print(results.params)
# подробная диагностическая информация о модели
print(results.summary())

# Оценивание процессов с временными рядами
init_x = 4
import random
values = [init_x, init_x]
N = 1000
b0 = 0.8
b1 = -0.4
noise = dnorm(0, 0.1, N)
for i in range(N):
    new_x = values[-1] * b0 + values[-2] * b1 + noise[i]
values.append(new_x)
# аппроксимировать модель
MAXLAGS = 5
model = sm.tsa.AR(values)
results = model.fit(MAXLAGS)
# оценках параметров(сначала свободный член, затем оценки двух первых лагов)
print(results.params)
