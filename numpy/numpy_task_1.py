# https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises_with_hints_with_solutions.md
# https://github.com/Yorko/mlcourse.ai/blob/master/jupyter_russian/topic01_pandas_data_analysis/lesson1_part0_numpy.ipynb
import numpy as np

# Умножение матриц и списков
# def no_numpy_mult(first, second):
#     result = [[0 for x in range(len(first))] for y in range(len(second))]
#     for i in range(len(first)):
#         for j in range(len(second[0])):
#             for k in range(len(second)):
#                 result[i][j] += first[i][k] * second[k][j]
#     return result
# def numpy_mult(first, second):
#     result = first @ second#YOUR CODE: create np.array
#     return result

# Сумма четных элементов на главной диагонали квадратной матрицы.
# def diag_2k(a):
#     n = np.diagonal(a, offset=0, axis1=0, axis2=1)
#     b = [x for x in n if x % 2 == 0]
#     if len(b) > 0:
#         result = np.array(sum(b))
#         return result
#     else:
#         return 0

# На вход подаётся 2 набора целых чисел.
# Создайте вектор V такой, что он будет содержать числа из 1 набора,
# делящиеся нацело на предпоследнее число из 2 набора и разделённые на это число.
# Если таких чисел не найдётся, то вектор V будет пустым (т.е. не будет содержать элементов).
# nn = input()
# mm = input()
# n = [int(x) for x in nn.split(', ')]
# m = [int(x) for x in mm.split(', ')]
# bb = []
# for i in n:
#     if i % m[-2] == 0:
#         bb.append(int(i / m[-2]))
# V = np.array(bb, dtype=float)

# Создайте массив класса np.ndarray ширины 4 и высоты 3 с двойками на главной диагонали и
# единицами на первой диагонали над главной
# print(2 * np.eye(3, 4) + np.eye(3, 4, 1))

# Превратите/перевернуть массив в вертикальный вектор.
# mat = 2 * np.eye(3, 4) + np.eye(3, 4, 1)
# vv = np.ndarray.flatten(mat)
# c = np.reshape(vv, (12, 1))

# На первой и третьей строке два целых положительных числа n и m - размерность матрицы.
# Вторая и четвертая строка элементы матрицы.
# Первые m чисел - первый ряд матрицы, числа от m+1 до 2⋅m - второй, и т.д.
# Напечатайте произведение матриц XY^TXY, игаче строку "matrix shapes do not match".
# x_shape = tuple(map(int, input().split()))
# X = np.fromiter(map(int, input().split()), np.int).reshape(x_shape)
# y_shape = tuple(map(int, input().split()))
# Y = np.fromiter(map(int, input().split()), np.int).reshape(y_shape)
# e = Y.T
# if x_shape[1] == y_shape[1]:
#     try:
#         b = np.dot(X, e)
#         print(b)
#     except ValueError:
#         print("matrix shapes do not match")
# else:
#     print("matrix shapes do not match")

# Считать данные из файла и посчитайте их средние значения.
# Дан адрес с csv-файлом, из которого нужно считать данные. Первая строка — названия столбцов, остальные строки — данные
# Создать вектор из средних значений вдоль столбцов входных данных.
# from urllib.request import urlopen#
# filename = input()
# f = urlopen(filename)
# sbux = np.loadtxt(f, dtype=float, skiprows=1, delimiter=",")
# v = sbux.mean(axis=0)

# В этой задаче нам даны 3 переменные: A1, A2, A3.
# Каждая содержит вектор с 2 координатами соответствующей вершины треугольника.
# Найдите площадь треугольника и выведите её на печать.
# Если все 3 точки лежат на одной прямой, то площадь треугольника равна 0.
# import math
# a = A1 - A2
# b = A2 - A3
# c = A3 - A1
# aa = (a[0] ** 2 + a[1] ** 2) ** 0.5
# bb = (b[0] ** 2 + b[1] ** 2) ** 0.5
# cc = (c[0] ** 2 + c[1] ** 2) ** 0.5
# # print(aa)
# # print(bb)
# # print(cc)
# p = (aa + bb + cc) / 2
# s = math.sqrt(p * (p - aa) * (p - bb) * (p - cc))

# Создайте в переменной Z Numpy вектор из нулей длины n.
# n = int(input())
# Z = np.zeros(n)

# Посчитайте размер матрицы Z в байтах и выведите его на печать.
# print(Z.nbytes)

# Считайте 2 числа:
# n - размер Numpy вектора
# x - координата элемента вектора, который должен быть равен 1.
# Остальные элементы вектора должны быть равны 0.
# nn = input()
# mm = input()
# Z = np.zeros(int(nn))
# Z[int(mm)] = 1

# Считайте 2 числа n, m.
# Создайте вектор Z состоящий из чисел от n до m с шагом 1.
# nn = input()
# mm = input()
# Z = np.arange(int(nn), int(mm) + 1)

# "Развернуть" вектор/матрицу.
# Z = Z[::-1]

# Считайте 3 числа:
# n - количество элементов матрицы
# m и l - размеры матрицы (число строк и столбцов соответственно)
# Заполните матрицу Z числами от 0 до n-1 по порядку (сперва строки, потом столбцы).
# Гарантируется, что m*l = n, т.е. все элементы "влезут" в матрицу и не останется пустых мест.
# nn = input()
# mm = input()
# mm_s = mm.split(" ")
# Z = np.arange(int(nn)).reshape(int(mm_s[0]), int(mm_s[1]))

# индексы ненулевых элементов.
# NonZerros = np.nonzero(Z)

# Создайте единичную матрицу размера n, сохраните результат в переменную Z.
# n = input()
# Z = np.eye(int(n))

# Считайте 3 числа: n, m, l.
# Зафиксируйте значение генератора случайных чисел Numpy с помощью numpy.random.seed(42)
# Создайте матрицу n*m*l из случайных чисел (от 0 до 1) и сохраните результат в переменную Z.
# n = input().split(" ")
# x = np.random.seed(42)
# Z = np.random.rand(int(n[0]), int(n[1]), int(n[2]))

# Считайте 2 числа: n, m.
# Зафиксируйте значение генератора случайных чисел Numpy с помощью numpy.random.seed(42)
# Создайте матрицу n*m из случайных чисел (от 0 до 1).
# Выведите на печать значение минимального и максимального чисел в получившейся матрице (каждое с новой строки).
# n = input().split(" ")
# x = np.random.seed(42)
# Z = np.random.rand(int(n[0]), int(n[1]))
# print(np.min(Z))
# print(np.max(Z))

# Считайте 2 числа: n, m.
# Зафиксируйте значение генератора случайных чисел Numpy с помощью numpy.random.seed(42)
# Создайте матрицу n*m из случайных чисел (от 0 до 1).
# Выведите на печать значение среднего для всей матрицы.
# n = input().split(" ")
# x = np.random.seed(42)
# Z = np.random.rand(int(n[0]), int(n[1]))
# print(np.mean(Z))

# Считайте 2 числа: n, m.
# Зафиксируйте значение генератора случайных чисел Numpy с помощью numpy.random.seed(42)
# Создайте матрицу n*m из случайных чисел (от 0 до 1).
# Найдите среднее значение для каждого из столбцов.
# Выведите на печать значение минимального и максимального среднего по столбцам (каждое с новой строки).
# n = input().split(" ")
# x = np.random.seed(42)
# Z = np.random.rand(int(n[0]), int(n[1]))
# mean_ = np.mean(Z, axis=0)
# min = np.min(mean_)
# max = np.max(mean_)
# print(min)
# print(max)

# Считайте 2 числа: n, m.
# Создайте матрицу размера n*m такую что:
# На границе матрицы будут стоять 1
# Внутри матрицы будут стоять 0
# n = input().split(" ")
# Z = np.ones((int(n[0]), int(n[1])))
# Z[1:-1, 1:-1] = 0

# Имеется матрица Z. Добавьте вокруг имеющихся значений матрицы "забор" из 0.
# Z = np.pad(Z, [(1, ), (1, )], mode='constant')

# Считайте число n. Создайте диагональную матрицу размера n*n. На главной диагонали должны быть числа от 1 до n.
# Сохраните матрицу в переменную Z.
# n = input()
# m = np.arange(1, int(n) + 1)
# Z = np.diag(m)

# Считайте 2 числа: n, m.
# Создайте матрицу размера n*m и "раскрасьте" её в шахматную раскраску.
# Ячейка с координатами (0, 0) всегда "чёрная" (т.е. элемент (0, 0) равен 0).
# n = input().split(" ")
# Z = np.zeros((int(n[0]),(int(n[1]))), dtype=float)
# Z[1::2,::2] = 1
# Z[::2,1::2] = 1

# Даны: индекс i and массив Z. Определите "координаты" элемента с индексом i в Z.
# Z = np.unravel_index(i, Z.shape)
# print(Z)

# Переменные A и B содержат по numpy вектору. Найдите их скалярное произведение и сохраните в переменную Z.
# Z = np.dot(A, B)

# 2 матрицы: A и B.
# Если найти произведение матриц невозможно, то запишите в Z строку: Упс! Что-то пошло не так...
# try:
#     Z = A @ B
# except ValueError:
#     Z = "Упс! Что-то пошло не так..."

# Поменяйте знак всех чисел из интервала (3; 9), хранящихся в векторе Z.
# Z[(3 < Z) & (Z < 9)] *= -1

# Bектор A содержит float числа как больше, так и меньше нуля.
# Округлите их до целых и результат запишите в переменную Z. Округление должно быть "от нуля", т.е.:
# положительные числа округляем всегда вверх до целого
# отрицательные числа округляем всегда вниз до целого
# 0 остаётся 0
# Z = np.copysign(np.ceil(np.abs(A)), A)

# Даны 2 вектора целых чисел A и B.
# Найдите числа, встречающиеся в обоих векторах и составьте их по возрастанию в вектор Z.
# Если пересечений нет, то вектор Z будет пустым.
# Z = np.intersect1d(A, B)

# Отключите вывод всех ошибок (это довольно опасное поведение).
# Z = np.seterr(all='ignore')

# Составьте список (numpy array) дат с шагом в 1 день от начала до окончания отсчёта (последний день не включается).
# Результат должен представлять из себя список дат в формате ISO.
# n = input()
# m = input()
# Z = np.arange(n, m, dtype='datetime64[D]')

# 1-я строка содержит 2 натуральных числа: n, m
# 2-я строка содержит число k
# Создайте матрицу размера n*m такую, что каждая её строка содержит числа от k до k+m-1 (с шагом 1).
# nn = input()
# n = nn.split(" ")
# m = input()
# f = np.arange(int(m), int(m) + int(n[1]), 1)
# f = f.astype("float64")
# Z = np.array([f,] * int(n[0]))

# 1-я строка содержит 2 натуральных числа: n, m
# 2-я строка содержит число k
# Создайте матрицу размера n*m такую, что каждый её столбец содержит числа от k до k+n-1 (с шагом 1).
# nn = input()
# n = nn.split(" ")
# m = input()
# f = np.arange(int(m), int(m) + int(n[0]), 1)
# f = f.astype("float64")
# Z = np.array([f,] * int(n[1])).T

# Составьте список (numpy.array) из n точек на отрезке (start, stop),
# попарно равноудалённых друг от друга и от концов отрезка (т.е. разделите отрезок на n+1 часть).
# Округлите значения точек до 3 знака после запятой.
# n = input()
# m = input()
# k = input()
# b = np.linspace(float(n), float(m), num=int(k) + 1, endpoint=False)[1:]
# Z = np.around(b, decimals=3)

# Составьте список из n точек на отрезке [start, stop] в геометрической прогрессии, включая start и stop.
# Округлите значения точек до 3 знака после запятой.
# n = input()
# m = input()
# k = input()
# b = np.geomspace(float(n), float(m), num=int(k))
# Z = np.around(b, decimals=3)

# С помощью Numpy сгенерируйте n случайных чисел из интервала (0, 1) с фиксированным seed,
# отсортируйте их по возрастанию и сохраните в переменную Z.
# n = input()
# m = input()
# np.random.seed(int(n))
# b = np.random.uniform(low=0.0, high=1.0, size=(int(m),))
# Z = np.sort(b)
