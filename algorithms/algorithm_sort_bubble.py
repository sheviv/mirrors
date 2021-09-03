# Алгоритм сортировки пузырьком(O(N**2))
# сравниваемые элемент в цикле поднимается/опускается и в следующем цикле не учитывается
a = [7, 5, -8, 0, 10, 1]
N = len(a)      # число элементов в списке
for i in range(0, N-1):     # N-1 итераций работы алгоритма
    for j in range(0, N-1-i):   # проход по оставшимся не отсортированным парам массива
        if a[j] > a[j+1]:
            a[j], a[j+1] = a[j+1], a[j]
# print(a)

# //////////////////////////////
def bubble_sort(nums):
    # Устанавливаем swapped в True, чтобы цикл запустился хотя бы один раз
    swapped = True
    while swapped:  # прерывается, когда элементы ни разу не меняются местами
        swapped = False
        for i in range(len(nums) - 1):
            if nums[i] > nums[i + 1]:
                # Меняем элементы
                nums[i], nums[i + 1] = nums[i + 1], nums[i]
                # Устанавливаем swapped в True для следующей итерации
                swapped = True
    return nums
# Проверяем, что оно работает
random_list_of_nums = [5, 2, 1, 8, 4]
print(bubble_sort(random_list_of_nums))
