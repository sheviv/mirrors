#!/usr/bin/python3

import os
"""
os
"""
# вывод содержимого каталога
os.listdir(".")
# переименнование файла
os.rename("1", "2")
# изменение прав доступа
os.chmod("my", 900)
# создание каталога
os.mkdir('/asd/zxc/dfg')
# рекурсивное создание каталога
os.makedirs('/asd/zxc')
# удаление файла
os.remove('qwe.py')
# удаление отдельного каталога
os.rmdir('/asd/fgh')
# удаление дерева каталогов
os.removedirs('/asd/zxc')
#  получение статистики файла/каталога
os.stat('asd')
os.stat_result()


"""
os.path
"""
# текущий каталог
cur = os.getcwd()
# конечный уровень пути
os.path.split(cur)
# родительский путь
os.path.dirname(cur)
# название конечного каталога
os.path.basename(cur)

