#!/usr/bin/python3
import logging
import threading
import time

# запустить отдельный поток
# def thread_function(name):
#     logging.info("Thread %s: starting", name)
#     time.sleep(2)
#     logging.info("Thread %s: finishing", name)
# if __name__ == "__main__":
#     format = "%(asctime)s: %(message)s"
#     logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
#     create threading
#     launch thread_function function and get args (1,) - only tuple
#     x = threading.Thread(target=thread_function, args=(1,))
#     x.start()
#     x.join()

# threading demons - процесс, который работает в фоновом режиме.
# join() - указать одному потоку дождаться завершения другого потока


"""
Работа с несколькими потоками
"""
# def thread_function(name):
#     logging.info("Thread %s: starting", name)
#     time.sleep(2)
#     logging.info("Thread %s: finishing", name)
# if __name__ == "__main__":
#     format = "%(asctime)s: %(message)s"
#     logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
#     threads = list()
#     for index in range(3):
#         logging.info("Main    : create and start thread %d.", index)
#         x = threading.Thread(target=thread_function, args=(index,))
#         threads.append(x)
#         x.start()
#     for index, thread in enumerate(threads):
#         logging.info("Main    : before joining thread %d.", index)
#         thread.join()
#         logging.info("Main    : thread %d done", index)


"""
ThreadPoolExecutor - способ запустить группу потоков
"""
import concurrent.futures
# def thread_function(name):
#     logging.info("Thread %s: starting", name)
#     time.sleep(2)
#     logging.info("Thread %s: finishing", name)
# if __name__ == "__main__":
#     format = "%(asctime)s: %(message)s"
#     logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
#     # with для управления созданием и удалением пула
#     # max_workers - сколько рабочих потоков
#     with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
#         #.map() для пошагового прохождения итерируемой объкта(каждый поток в пул)
#         executor.map(thread_function, range(3))
#         # конец блока with заставляет выполнять .join() для каждого из потоков


"""
Условия гонки (Race Conditions) - два или более потока обращаются к общему фрагменту данных или ресурсу
"""
# class FakeDatabase:
#     def __init__(self):
#         self.value = 0
#     def update(self, name):
#         """
#         имитирует чтение из бд, делается вычисления
#         и записывает новое значение в бд
#         """
#         logging.info("Thread %s: starting update", name)
#         local_copy = self.value
#         local_copy += 1
#         time.sleep(0.1)
#         self.value = local_copy
#         logging.info("Thread %s: finishing update", name)
# if __name__ == "__main__":
#     format = "%(asctime)s: %(message)s"
#     logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
#     database = FakeDatabase()
#     logging.info("Testing update. Starting value is %d.", database.value)
#     with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
#         for index in range(2):
#             # позволяет передавать как позиционные, так и именованные аргументы функции
#             # .submit(function, *args, **kwargs)
#             executor.submit(database.update, index)
#     logging.info("Testing update. Ending value is %d.", database.value)


"""
Lock — только один поток за раз может использовать Lock(другой поток должен ждать)
"""
# my_lock.acquire() - получить блокировку потоку
# or
# with - для управления созданием и удалением пула
class FakeDatabase:
    def __init__(self):
        self.value = 0
        # self._lock - в разблокированном состоянии
        self._lock = threading.Lock()
    def locked_update(self, name):
        logging.info("Thread %s: starting update", name)
        logging.debug("Thread %s about to lock", name)
        # блокируется и освобождается оператором
        with self._lock:
            logging.debug("Thread %s has lock", name)
            local_copy = self.value
            local_copy += 1
            time.sleep(0.1)
            self.value = local_copy
        logging.info("Thread %s: finishing update", name)
if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    database = FakeDatabase()
    logging.info("Testing update. Starting value is %d.", database.value)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for index in range(2):
            # позволяет передавать как позиционные, так и именованные аргументы функции
            # .submit(function, *args, **kwargs)
            executor.submit(database.locked_update, index)
    logging.info("Testing update. Ending value is %d.", database.value)
