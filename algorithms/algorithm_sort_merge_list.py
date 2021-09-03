# Сортировка слиянием O(nlogn)
# функция слияния двух отсортированных списков
def merge_list(a, b):
    c = []
    N = len(a)
    M = len(b)
    i = 0
    j = 0
    while i < N and j < M:
        if a[i] <= b[j]:
            c.append(a[i])
            i += 1
        else:
            c.append(b[j])
            j += 1
    c += a[i:] + b[j:]
    return c
# функция деления списка и слияния списков в общий отсортированный список
def split_and_merge_list(a):
    N1 = len(a) // 2
    a1 = a[:N1]     # деление массива на два примерно равной длины
    a2 = a[N1:]
    if len(a1) > 1:  # если длина 1-го списка больше 1, то делим дальше
        a1 = split_and_merge_list(a1)
    if len(a2) > 1:  # если длина 2-го списка больше 1, то делим дальше
        a2 = split_and_merge_list(a2)
    return merge_list(a1, a2)   # слияние двух отсортированных списков в один
a = [9, 5, -3, 4, 7, 8, -8]
a = split_and_merge_list(a)
# print(a)

# //////////////////////////////
# Список разбивается пополам, пока не останутся единичные элементы.
# Соседние элементы становятся отсортированными парами(пары объединяются и сортируются с другими).
def merge(left_list, right_list):
    sorted_list = []
    left_list_index = right_list_index = 0
    # Длина списков часто используется, поэтому создадим переменные для удобства
    left_list_length, right_list_length = len(left_list), len(right_list)
    for _ in range(left_list_length + right_list_length):
        if left_list_index < left_list_length and right_list_index < right_list_length:
            # Сравниваем первые элементы в начале каждого списка
            # Если первый элемент левого подсписка меньше, добавляем его в отсортированный массив
            if left_list[left_list_index] <= right_list[right_list_index]:
                sorted_list.append(left_list[left_list_index])
                left_list_index += 1
            # Если первый элемент правого подсписка меньше,
            # добавляем его в отсортированный массив
            else:
                sorted_list.append(right_list[right_list_index])
                right_list_index += 1
        # Если достигнут конец левого списка, элементы правого списка
        # добавляем в конец результирующего списка
        elif left_list_index == left_list_length:
            sorted_list.append(right_list[right_list_index])
            right_list_index += 1
        # Если достигнут конец правого списка, элементы левого списка
        # добавляем в отсортированный массив
        elif right_list_index == right_list_length:
            sorted_list.append(left_list[left_list_index])
            left_list_index += 1
    return sorted_list
def merge_sort(nums):
    # Возвращаем список, если он состоит из одного элемента
    if len(nums) <= 1:
        return nums
    # Для того чтобы найти середину списка, используем деление без остатка
    # Индексы должны быть integer
    mid = len(nums) // 2
    # Сортируем и объединяем подсписки
    left_list = merge_sort(nums[:mid])
    right_list = merge_sort(nums[mid:])
    # Объединяем отсортированные списки в результирующий
    return merge(left_list, right_list)
# Проверяем, что оно работает
random_list_of_nums = [120, 45, 68, 250, 176]
random_list_of_nums = merge_sort(random_list_of_nums)
print(random_list_of_nums)
