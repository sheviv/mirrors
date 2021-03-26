import numpy as np
import pandas as pd

# Создайте Dataframe из словаря data. В качестве подписей строк используйте список labels.
# df = pd.DataFrame(data, index=labels)

# Открытие файла "name.csv", который внутри разделен знаком ";",
# считать столбы 'start_at', 'end_at' как дату("parse_dates")
# df = pd.read_csv('name.csv', sep=";", parse_dates=['start_at', 'end_at'])

# Выведите на печать содержимое ячейки
# print(df[col][row])

# Типы колонок
# df.dtypes

# Выведите на печать:
# число непустых (не null) значений в колонке 'age'
# 75% квантиль для значений в колонке 'age'
# ad = df[df.age != np.nan].count()
# print(float(ad['age']))
# s = df["age"].quantile(.75)
# print(s)

# Посчет "male" в столбике "Sex"
# df[df['Sex'] == 'male'].count()

# Отсеять ячейки, у которых нет никакого значения:
# df.notna()

# Показать только уникальные значения:
# unique()

# Выведите на печать n первых строки:
# ad = df.head(n)

# Показать последние n значений:
# df.tail(n)

# Создание новой колонки:
# df['new_column'] = column # new_column - новая колонка, column - может быть чуть ли не любым значением

# Сортировка значений:
# df.sort_values(by='column_name') # где column_name - название столбца

# Выведите на печать строки с индексами 0, 2, 3.
# ad = df.iloc[[0, 2, 3]]

# Выведите на печать только столбцы 'name' и 'age'
# ad = df[["name", "age"]]

# Выведите на печать только столбцы 'name' и 'age' И строки 0, 2, 3
# dg = df.iloc[[0, 2, 3]]
# ad = dg[["name", "age"]]

# Данные больше N в "age" столбце.
# dg = df[df.age > float(N)]

# данные, в которых в графе 'age' стоит null
# dg = df[df['age'].isnull()]

# данные, в которых в графе 'age' не стоит null
# df[df['age'].notnull()]

# (имя столбца = 'animal'), (значение = 'cat'), (имя столбца = 'age') которых меньше 3.
# filter_names = ["animal", "age"]
# filter_values = ["cat", 3]
# dg = df[(df[filter_names[1]] < filter_values[1]) & (df[filter_names[0]] == filter_values[0])]

# интервале между числами age_between (включая границы).
# age_between = [2, 4]
# dg = df[(df["age"] >= age_between[0]) & (df["age"] <= age_between[1])]

# Заполнение нового столбца 'temp_C' данными из столбца 'temp' функции "temp_to_celcius"
# df['temp_C'] = temp_to_celcius(df['temp'])

# Поставить андерскор(нижнее подчеркивание) "_" вместо пробелов и привести к нижнему регистру названия колонок:
# def repl(col):
#     new_col = col.replace(" ", "_").lower()
#     return new_col
# keys = []
# for i in df.columns:
#     keys.append(i)
# values = [repl(x) for x in keys]
# dictionary = dict(zip(keys, values))
# df.rename(columns=dictionary, inplace=True)

# Переменная index содержит строку ИЛИ число (индекс)
# Увеличьте значение возраста в строке индексом равным index на 1.
# if type(index) == str:
#     pf = df['age'][index]
#     df.loc[index, 'age'] = pf + 1
#     print(df)
# if type(index) == int:
#     pf = df['age'][index]
#     df.loc[index, 'age'] = pf + 1
#     print(df)

# индексация относится к столбцам, срезы относятся к строкам:
# data['Florida':'Illinois']
#           area   pop     density
# Florida 170312 19552860 114.806121
# Illinois 149995 12882135 85.883763

# операции маскирования интерпретируются построчно,а не по столбцам:
# data[data.density > 100]
#           area     pop    density
# Florida 170312 19552860 114.806121
# New York 141297 19651127 139.076746

# агрегирующие функции, игнорирующие пропущенные значения NaN:
#np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2)

# выявление пустых значений: isnull() и notnull().
# data = pd.Series([1, np.nan, 'hello', None])
# data.isnull()

# методы dropna() (отбрасывающие NA-значения) и fillna() (заполняющий NA значения)
# dropna() отбрасывает все строки, в которых присутствует хотя бы одно пустое значение:
# data.dropna()

# how='all', при нем будут отбрасываться только строки/столбцы, все значения в которых пустые:
# df.dropna(axis='columns', how='all')

# с помощью параметра thresh минимальное количество непустых значений для строки/столбца, при котором он не отбрасывается:
# df.dropna(axis='rows', thresh=3)

# Заполнение пустых значений.
# Можно заполнить NA-элементы фиксированным значением(пример: нулями):
# data.fillna(0)

# параметр заполнения по направлению «вперед», копируя предыдущее значение в следующую ячейку:
# # заполнение по направлению «вперед»
# data.fillna(method='ffill')
# параметр заполнения по направлению «назад», копируя следующее значение в предыдущую ячейку:
# заполнение по направлению «назад»
# data.fillna(method='bfill')
# Для объектов DataFrame:
# df.fillna(method='ffill', axis=1)

# Мультииндексированный объект Series
# index = [('California', 2000), ('California', 2010), ('New York', 2000), ('New York', 2010),
# ('Texas', 2000), ('Texas', 2010)]
# index = pd.MultiIndex.from_tuples(index)
# pop = pop.reindex(index)
# California 2000 33871648
#            2010 37253956
# New York   2000 18976457
#            2010 19378102
# Texas      2000 20851820
#            2010 25145561

# Метод unstack() может преобразовать мультииндексный объект Series в индексированный обычным образом объект DataFrame:
# pop_df = pop.unstack()
#             2000     2010
# California  33871648 37253956
# New York    18976457 19378102
# Texas       20851820 25145561

# метод stack() выполняет противоположную операцию:
# pop_df.stack()
# California 2000 33871648
#            2010 37253956
# New York   2000 18976457
#            2010 19378102
# Texas      2000 20851820
#            2010 25145561

# Методы создания мультииндексов.
# df = pd.DataFrame(np.random.rand(4, 2),
# index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
# columns=['data1', 'data2'])
#     data1    data2
# a 1 0.554233 0.356072
#   2 0.925244 0.219474
# b 1 0.441759 0.610054
#   2 0.171495 0.886688

# Иерархические индексы и столбцы
# index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]], names=['year', 'visit'])
# columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']], names=['subject', 'type'])
# Создаем имитационные данные
# data = np.round(np.random.randn(4, 6), 1)
# data[:, ::2] *= 10
# data += 37
# Создаем объект DataFrame
# health_data = pd.DataFrame(data, index=index, columns=columns)
# subject    Bob      Guido      Sue
# type       HR Temp   HR  Temp  HR  Temp
# year visit
# 2013  1   31.0 38.7 32.0 36.7 35.0 37.2
#       2   44.0 37.7 50.0 35.0 29.0 36.7
# 2014  1   30.0 37.4 39.0 37.8 61.0 36.9
#       2   47.0 37.8 48.0 37.3 51.0 36.5
# информация только об этом человеке(ячейке)
# health_data['Guido']
# type      HR  Temp
# year visit
# 2013  1  32.0 36.7
#       2  50.0 35.0
# 2014  1  39.0 37.8
#       2  48.0 37.3

# Обращение к отдельным элементам путем индексации с помощью нескольких термов:
# health_data['2013']['1']
# 32.0 36.7

# срез, если мультииндекс отсортирован
# health_data.loc['2013':'New 2014']

# срез явным образом с помощью встроенной функции Python slice() или использовать объект IndexSlice,
# idx = pd.IndexSlice
# health_data.loc[idx[:, 1], idx[:, 'HR']]
# subject   Bob Guido  Sue
# type      HR   HR    HR
# year visit
# 2013   1  31.0 32.0 35.0
# 2014   1  30.0 39.0 61.0

# конкатенация с помощью метода pd.concat
#ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
# ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])
# pd.concat([ser1, ser2])
# 1 A
# 2 B
# 3 C
# 4 D
# 5 E
# 6 F

# Дублирование индексов при конкатенации
# x = make_df('AB', [0, 1])
# y = make_df('AB', [2, 3])
# y.index = x.index # Дублируем индексы!
# pd.concat([x, y]))
#    A  B
# 0 A0 B0
# 1 A1 B1
# 0 A2 B2
# 1 A3 B3

# Соединения «один-к-одному». «Объединение наборов данных: конкатенация и добавление в конец», в один объект DataFrame
# df3 = pd.merge(df1, df2)

# Соединения «многие-к-одному». соединения, один из двух ключевых столбцов содержит дублирующиеся значения.
# объекте DataFrame имеется дополнительный столбец с информацией с повторением информации в одном или нескольких местах
# pd.merge(df3, df4)
# Соединения «многие-ко-многим», столбец ключа как в левом, так и в правом массивах содержит повторяющиеся значения

# Пересекающиеся названия столбцов:
# pd.merge(df8, df9, on="name", suffixes=["_L", "_R"])
#    name rank_L rank_R
# 0   Bob   1     3
# 1   Jake  2     1
# 2   Lisa  3     4
# 3   Sue   4     2

# индексация по времени
# index = pd.DatetimeIndex(['2014-07-04', '2014-08-04', '2015-07-04', '2015-08-04'])
# data = pd.Series([0, 1, 2, 3], index=index)
# 2014-07-04 0
# 2014-08-04 1
# 2015-07-04 2
# 2015-08-04 3

# указать год, чтобы получить срез всех данных за 2015 год:
# data['2015']

# pd.to_datetime() отдельной даты она возвращает Timestamp, при передаче ряда дат
# dates = pd.to_datetime([datetime(2015, 7, 3), '4th of July, 2015', '2015-Jul-6', '07-07-2015', '20150708'])
# указав код для периодичности интервала - код 'D', периодичность интервала — один день:
# dates.to_period('D')

# создаем диапазон часовых меток даты/времени:
# pd.date_range('2015-07-03', periods=8, freq='H')

# создать смещение в один рабочий день следующим образом:
# from pandas.tseries.offsets import BDay
# pd.date_range('2015-07-01', periods=5, freq=BDay())

# DataFrame.eval() - создания нового столбца 'D' и присваивания ему значения, вычисленного на основе других столбцов:
# df.eval('D = (A + B) / C', inplace=True)

# Увеличьте значение возраста во всех строках на 1.
# df["age"] += 1

# Переменная group_by содержит имя колонки по которой производится группировка.
# Найдите средние значение возраста по всем записям,
# c = df.groupby(group_by)["age"].mean()

# Добавьте новую строку (с индексом new_index и значениями new_data) и удалите одну из старых (del_index)
# df.loc[new_index] = new_data
# df.drop([del_index], inplace=True)

# Показать значения различающихся значений из столбцов "reserved_room_type", "assigned_room_type"
# query() - метод query принимает запрос в виде строки
# df.query("reserved_room_type != assigned_room_type")
# df[["reserved_room_type", "assigned_room_type"]]

# Подсчет суммы "sum" колонки "money" по названию другой колонки "tittle"
# agrgragate - подсчет/вычет(работа с данными) и т.д.
# as_index=False - вернет привычные индексы
# sort_values() - сортировка знакчений(ascending - сортировка по значению в столбце)
# df.groupby(["tittle"], as_index=False).aggregate({"money": "sum"}).sort_values("money", ascending=False)

# Группировка данных по столбцу "borough", суммируя значения из 'pickups'.
# df.groupby('borough').agg({'pickups': 'sum'})

# Группировка данных по столбцу "borough", суммируя значения из 'pickups' и поиск минимального/максимального значения
# df.groupby('borough').agg({'pickups': 'sum'}).idxmin() # idxmax()

# Поиск среднего 'mean' значения в 'pickups' используя "borough" и "hday":
# df.groupby(["borough", "hday"]).agg({'pickups': 'mean'})

# Сохранить в .csv файл полученный таблицу/dataframe
# index=False - сбрасываем индексы
# money = df.groupby(["tittle"], as_index=False).aggregate({"money": "sum"}).sort_values("money", ascending=False)
# money.to_csv("money.csv", index=False)

# Сохранить в названии файла "file_name" сегоднящнюю дату "cv"
# from datetime import datetime
# money = df.groupby(["tittle"], as_index=False).aggregate({"money": "sum"}).sort_values("money", ascending=False)
# cv = datetime.today().strftime("%Y-%m-%d")
# file_name = f"money_{cv}.csv"
# money.to_csv(file_name, index=False)

# Найдите количество записей каждого типа, сгруппированным по значению в колонке group_by.
# cv = df[group_by].value_counts()

# Посчитайте частоту встречаемости "driver_score", перевод в проценты и округление до 2 знаков, сбросить индексы
# Переименуйте колонки "index": "driver_score", "driver_score": "percentage"
# Отсортировать "driver_score" в порядке возрастания
# driver_score_counts = taxi["driver_score"].\
#     value_counts(normalize=True).\
#     mul(100).\
#     round(2).\
#     reset_index().\
#     rename(columns={"index": "driver_score", "driver_score": "percentage"}).\
#     sort_values("driver_score")

# Подсчет данных по столбцу "arrival_date_year" и "arrival_date_month"
# для значения "City Hotel" из столбца "hotel" при условии "is_canceled" = 1(True)
# df.query("hotel == 'City Hotel' & is_canceled == 1").groupby('arrival_date_year')["arrival_date_month"].value_counts()

# Числовые характеристики(пример: "mean") трёх колонок: "adults", "children" и "babies"
# df.agg({'adults': "mean", "children": "mean", "babies": "mean"})

# Переменная sort_by содержит список из 2 строк - имён столбцов по которым необходимо провести сортировку.
# Отсортируйте сперва по уменьшению 1 поля из списка sort_by, а при равенстве значений по увеличению 2.
# sort_by = ["age", "visits"]
# cv = df.sort_values([sort_by[0], sort_by[1]], ascending=[0, 1])

# Переменная column содержит имя колонки, содержащей строковые значения "yes", "No", либо числовые 1 или 0.
# "yes" и 1 на True
# "no" и 0 на False
# column = "on vacation"
# df[column] = df[column].map({'yes': True, 1: True, 0: False, 'no': False})

# column содержит имя колонки, значения в которой надо модифицировать.
# old_value и new_value содержат старое и новое значения, соответственно.
# Замените все старые значения на новые в соответсвующей колонке.
# df[column] = df[column].replace(old_value, new_value)

# Создать столбец "total_kids" из "children" и "babies" и сгруппировать с "hotel" и найти в нем среднее
# В качестве ответа укажите наибольшее среднее total_kids, округлив до 2 знаков после точки.
# df["total_kids"] = df["children"] + df["babies"]
# df.groupby("hotel").agg({"total_kids": "mean"}).round(2)