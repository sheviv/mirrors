#!/usr/bin/env python

from collections import OrderedDict
import commands
import os
import sqlite3


# Подключение текста меню к командам бизнес-логики
class Option:
    def __init__(self, name, command, prep_call=None):
        self.name = name  # Имя, показываемое в меню
        self.command = command  # Экземпляр выполняемой команды
        self.prep_call = prep_call  # Вызывается перед выполнением команды

    def choose(self):
        """
        будет вызван, когда вариант действия выбран из меню
        Returns:
        """
        # Вызывает подготовительный шаг, если он указан
        data = self.prep_call() if self.prep_call else None
        message = self.command.execute(data) if data else self.command.execute()
        print(message)

    def __str__(self):
        return self.name


def clear_screen():
    """
    Очистка экрана непосредственно перед печатью меню
    Returns:
    """
    clear = 'cls' if os.name == 'nt' else 'clear'
    os.system(clear)


def option_choice_is_valid(choice, options):
    """
    Вариант является допустимым, если буква совпадает с одним из ключей в словаре options
    Args:
        choice:
        options:
    Returns:
    """
    return choice in options or choice.upper() in options


def get_option_choice(options):
    # Получает от пользователя первоначальный вариант
    choice = input('Choose an option: ')
    # Пока вариант пользователя остается допустимым, продолжать предлагать ему ввести данные
    while not option_choice_is_valid(choice, options):
        print('Invalid choice')
        choice = input('Choose an option: ')
    # Возвращает совпадающий вариант, когда сделан правильный выбор
    return options[choice.upper()]


def get_user_input(label, required=True):
    """
    Общая функция, которая предлагает пользователю ввести данные
    Args:
        label:
        required:
    Returns:
    """
    # Получение первоначальный ввод от пользователя
    value = input(f'{label}: ') or None
    # Ввод до тех пор, пока входные данные остаются пустыми
    while required and not value:
        value = input(f'{label}: ') or None
    return value


def get_new_bookmark_data():
    """
    Получение необходимых данных для добавления новой закладки
    Returns:
    """
    # закладки являются необязательными, поэтому не продолжает предлагать их ввести
    return {
        'title': get_user_input('Title'),
        'url': get_user_input('URL'),
        'notes': get_user_input('Notes', required=False),
    }

def get_bookmark_id_for_deletion():
    """
    Получение необходимой информации для удаления закладки
    Returns:
    """
    return get_user_input('Enter a bookmark ID to delete')



def print_options(options):
    """
    Детализация и вывод вариантов действий в меню
    Returns:
    """
    for shortcut, option in options.items():
        print(f'({shortcut}) {option}')
    print()


def loop():
    """
    Все, что происходит для каждой итерации цикла меню > опция > результат уходит сюда
    Returns:
    """
    clear_screen()

    options = OrderedDict({
        'A': Option('Add a bookmark', commands.AddBookmarkCommand(), prep_call=get_new_bookmark_data),
        'B': Option('List bookmarks by date', commands.ListBookmarksCommand()),
        'T': Option('List bookmarks by title', commands.ListBookmarksCommand(order_by='title')),
        'D': Option('Delete a bookmark', commands.DeleteBookmarkCommand(), prep_call=get_bookmark_id_for_deletion),
        'Q': Option('Quit', commands.QuitCommand()),
    })
    print_options(options)

    chosen_option = get_option_choice(options)
    clear_screen()
    chosen_option.choose()

    # Предлагает нажать ENTER и просматривает результат перед продолжением работы(_означает «неиспользуемое значение»)
    _ = input('Press ENTER to return to menu')


if __name__ == '__main__':
    """
    исключает случайное выполнение кода в модуле при импортировании модуля bark
    """
    commands.CreateBookmarksTableCommand().execute()
    # Повторяет в бесконечном цикле(пока пользователь не выберет вариант, соответствующий команде QuitCommand)
    while True:
        loop()


def for_listings_only():
    options = {
        'A': Option('Add a bookmark', commands.AddBookmarkCommand()),
        'B': Option('List bookmarks by date', commands.ListBookmarksCommand()),
        'T': Option('List bookmarks by title', commands.ListBookmarksCommand(order_by='title')),
        'D': Option('Delete a bookmark', commands.DeleteBookmarkCommand()),
        'Q': Option('Quit', commands.QuitCommand()),
    }
    print_options(options)
