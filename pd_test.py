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