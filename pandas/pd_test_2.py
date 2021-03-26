import numpy as np
import pandas as pd

# Соединение и конкатинация таблиц с обнулением индексов
# Log_1 = pd.read_csv('log_1.csv', sep=';')
# Log_2 = pd.read_csv('log_2.csv', sep=';')
# Log_1.append(Log_2, ignore_index=True)
# pd.concat((Log_1, Log_2), ignore_index=True)

# Добавление отдельной строки
# Log_2.append([{'ID':1, 'SOURCE':'user', 'DATE':'now'}])

# Объединение таблиц по столбцам:
# 1. Изменить ось(направление) объединения со строк (0) на столбцы (1):
# Log_2_success = pd.read...
# pd.concat((Log_2, Log_2_success), axis = 1)
# При дублировании, удалить столбец, просто сделав выборку нужных данных:
# pd.concat((Log_2, Log_2_success['SUCCESS']), axis = 1)

# Если данные перепутаны при соединении/ конкатенации таблиц.
# В параметр on передадется кортеж с именами ключей, в которых надо искать соответствие.
# функция берет строку из 1 таблицы, ищет в "ID", ищет строку с таким же значением в "ID" 2 таблицы и склеивает их.
# merge удаляет повторяющиеся столбцы "ID" + строки без "пар"
# pd.merge(Log_2, Log_2_success_unsorted, on=('ID'))

# Объединение 3 таблиц
# pd.merge(Log_1.append(Log_2), Log_2_success_unsorted, on=('ID'))

# Удаление повторяющихся столбцов, но оставить незаполненные.
# pd.merge(Log_1.append(Log_2), Log_2_success_unsorted, on=('ID'), how ='left')

# Объединить "data" с "df", по колонке "client_id"
# df.merge(data)

# Объединение таблиц, находя соответствие по индексам.
# 1. Объединяем таблицы(это не сработает, если имена колонок/индексов разные)
# pd.merge(Log_1_i.append(Log_2_i), Log_2_success_unsorted_i, on=('ID'), how ='left')
# 2. Подготовим колонку индексов в которой переименуем в "NEW_ID":
# Log_2_success_unsorted_i.index.names = ["NEW_ID"]
# left_index=True - для использования колонки индексов для объединения для левой таблицы (1-й аргумент)
# right_index=True - для использования колонки индексов для объединения для правой таблицы (2-й аргумент)
# pd.merge(Log_1_i.append(Log_2_i), Log_2_success_unsorted_i, how ='left', left_index=True, right_index=True)

# У одной таблицы мы хотим использовать индекс, а у другой колонку для объединения.
# Кроме указанных выше аргументов нужны 1 или оба аргумента:
# left_on - имя колонки из левой таблицы (1-й аргумент) для объединения
# right_on - имя колонки из правой таблицы (2-й аргумент) для объединения
# pd.merge(Log_1_i.append(Log_2_i), Log_2_success_unsorted_i, how ='left', left_on=('ID'), right_index=True)

# Связь один ко многим
# Нет однозначного соответствия строк одной таблицы другой
# orders.csv - информация по позициям в заказе (по 1 товару в строке, если в заказе несколько товаров,
# то в таблице будет несколько записей)
# Products.csv - выгрузка из справочника товаров (название, ID, цена и валюта)
# Не все товары из Products.csv фигурируют в orders.csv.
# Некоторые товары (id 86, 103, 104) купили более одного раза.
# orders = pd.read_csv('orders.csv', sep=";")
# products = pd.read_csv('Products.csv', sep=";")
# Указываем имена колонок по которым производим объединение:
# pd.merge(orders, products, how ='left', left_on=('ID товара'), right_on=('Product_ID'))
# Удалить строки с NaN в одной из колонок, можно объединением how='inner'
# (тогда строки из левой таблицы, которым не найдено соответствие в правой, так же будут исключены из результата).
# pd.merge(orders, products, how='inner', left_on=('ID товара'), right_on=('Product_ID'))

# 1) Создать колонку "brand_name" и заполнить ее данными колонки "brand_info"
# из функции "split_brand", которая считывает последнее слово в строке
# def split_brand(brand_name_data):
#     return brand_name_data.split(" ")[-1]
# df["brand_name"] = df["brand_info"].apply(split_brand)

# 2) Использование lambda функции для задачи выше
# df["brand_name"] = df["brand_info"].apply(lambds x: x.split(" ")[-1])