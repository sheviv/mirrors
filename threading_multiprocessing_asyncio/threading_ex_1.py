#!/usr/bin/python3
import logging
import threading
import time


"""
example
"""
# def myfunc(a, b):
#     print('сумма :',a + b)
# thr1 = threading.Thread(target = myfunc, args = (1, 2)).start()  # отдельный поток
# print('основной поток')


"""
threading.Thread() - создать новый поток, создав экземпляр класса Thread
"""
# threading.Thread(group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None)
"""
group - зарезервирована для будущего расширения при реализации класса ThreadGroup.
target - выполняется в потоке с помощью метода run().
name - имя потока, по умолчанию - «Thread-X», где X – десятичное число.
args - кортеж, в котором хранятся аргументы в вызываемую функцию.
kwargs - словарь, в котором хранятся аргументы, передаваемые в функцию.
daemon - устанавливает, является ли поток демоническим(работают в фоновом режиме).
"""



"""
start() - для запуска созданного потока
"""
# def myfunc(a, b): 
#     print('сумма :',a + b) 
# thr1 = threading.Thread(target = myfunc, args = (1, 2))  # пока не вызван метод start, myfunc - не будет запущен.
# thr1.start()


"""
join() - блокирует выполнение потока, который его вызвал, до тех пор пока не завершится поток, метод которого был вызван.
"""
# hr1.join(100) - ожидается завершение выполнения потока thr1 не более 100с.
# is_alive() - выполнился ли поток.
# def myfunc(a, b):
#     time.sleep(2.5)
#     print('сумма :', a + b)
# # demon=True - чтобы программа не дожидалась окончания выполнения функции
# thr1 = threading.Thread(target = myfunc, args = (1, 2), daemon=True)
# thr1.start()
# thr1.join(0.125)  # приостанавливаем основной поток на 0.125с
# # True - поток не закончил выполнение за 0.125с
# if thr1.is_alive():
#     print('поток не успел завершиться')
# else:
#     print('вычисления завершены')


"""
run() - операции, выполняемые потоком(когда явно создается экземпляр класса)
"""
# class Thr1(threading.Thread): # Создаём экземпляр потока Thread
#     def __init__(self, var):
#         threading.Thread.__init__(self)
#         self.daemon = True # Указываем, что этот поток - демон
#         self.var = var # это интервал, передаваемый в качестве аргумента

#     def run(self): # метод, который выполняется при запуске потока
#         num = 1
#         while True:
#             y = num*num + num / (num - 10) # Вычисляем функцию
#             num += 1
#             print("При num =", num, " функция y =", y) # Печатаем результат
#             time.sleep(self.var) # Ждём указанное количество секунд
# x = Thr1(0.9)
# x.start()
# time.sleep(2)


"""
is_alive() - проверяет выполняется ли поток в данный момент
"""
# while True:
#     if thr1.is_alive() == True: # Проверяем, выполняется ли поток демон
#         time.sleep(1) # Если да, ждем 1 секунду и проверяем снова
#     else:
#         break # Если нет, выходим из цикла и закрываем программу


"""
Остановка потока
1. В бесконечном цикле проверть True or False.
2. Не использовать функции, с блокировкой на длительное время. Использовать timeout.
"""
# stop = False
# def myfunc():
#     global stop
#     while stop == False:
#         pass
# thr1 = threading.Thread(target = myfunc) 
# thr1.start() 
# stop = True
# while thr1.is_alive() == True: 
#     pass
# print('поток завершился')


"""
Состояние гонки – ошибка при неправильном проектировании программы
(несколько потоков обращаются к одним и тем же данным)
"""
x = 5
# Thread 1:
if x == 5: # Поток 1 проверяет условие и считает его верным
    pass
# Thread 2:
x = 1 # Поток два изменяет значение переменной
# Thread 1:
print("При x = 5 функция 2*x =", 2 * x) # Поток один выполняет действие