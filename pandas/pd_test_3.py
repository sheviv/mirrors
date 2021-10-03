import pandas as pd
from pandas import Series, DataFrame


# Корреляция и ковариация
import pandas_datareader.data as web
all_data = {ticker: web.get_data_yahoo(ticker) for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOG']}
price = pd.DataFrame({ticker: data['Adj Close'] for ticker, data in all_data.items()})
volume = pd.DataFrame({ticker: data['Volume'] for ticker, data in all_data.items()})
returns = price.pct_change()
# Corr вычисляет корреляционную
# print(returns['MSFT'].corr(returns['IBM']))
# Cov вычисляет ковариацию
# print(returns['MSFT'].cov(returns['IBM']))
# corr и cov DataFrame возвращают полную корреляционную или ковариационную матрицу
# print(returns.corr())
# print(returns.cov())

# Обработка отсутствующих данных
# dropna
# fillna
# isnull
# notnull

# Фильтрация отсутствующих данных
from numpy import nan as NA
# data = Series([1, NA, 3.5, NA, 7])
# data.dropna() or data[data.notnull()]
# ///
# data = DataFrame([[1., 6.5, 3.], [1., NA, NA], [NA, NA, NA], [NA, 6.5, 3.]])
# cleaned = data.dropna()


# Восполнение отсутствующих данных
# Константа, подставляемая вместо отсутствующих значений:
# df.fillna(0)  # новый объект
# ///
# Модифицировать существующий объект
# _ = df.fillna(0, inplace=True)
# ///
# Передать среднее или медиану объекта в отсутствующие места
# data.fillna(data.mean())


# Преобразование данных
# Устранение дубликатов
# data.drop_duplicates()

# Замена значений
# data.replace(-999, np.nan)
# ///
# Дискретизация и раскладывание
# Разложить значения по ящикам
# ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
# bins = [18, 25, 35, 60, 100]  # от 18 до 25, от 26 до 35, от 35 до 60
# cats = pd.cut(ages, bins)
# ///
# Обнаружение и фильтрация выбросов
# превышающие 3 по абсолютной величине
# col[np.abs(col) > 3]
# ///
# срезать значения, выходящие за границы интервала от –3 до 3
# data[np.abs(data) > 3] = np.sign(data) * 3
