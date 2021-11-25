#!/usr/bin/env python

from collections import OrderedDict
import commands
import os
import sqlite3

class DatabaseManager:
    def __init__(self, database_filename):
        """
         Создает и сохраняет соединение с БД для последующего использования
        Args:
            database_filename:
        """
        self.connection = sqlite3.connect(database_filename)

    def __del__(self):
        """
        Закрывает соединение, когда дело сделано, на всякий случай
        Returns:
        """
        self.connection.close()

    def _execute(self, statement, values=None):
        """
        1) принять инструкцию в качестве аргумента;
        2) получить курсор из соединения с БД;
        3) выполнить инструкцию с помощью курсора (подробнее об этом чуть позже);
        4) вернуть курсор, который сохранил результат выполненной инструкции (если таковой имеется).
        Args:
            statement:
        Returns:
        """
        # Создает контекст транзакции БД
        with self.connection:
            # Создает курсор
            cursor = self.connection.cursor()
            # Использует курсор для выполнения инструкций SQL(исполняет инструкцию, предоставляя плейсхолдерам
            # все переданные внутрь значения)
            cursor.execute(statement, values or [])
            return cursor

    def create_table(self, table_name, columns):
        """
        Создание таблицы БД SQLite
        Args:
            table_name:
            columns:
        Returns:
        """
        # Конструирует определения столбцов с их типами и ограничениями
        columns_with_types = [
            f'{column_name} {data_type}'
            for column_name, data_type in columns.items()
        ]

        # Конструирует полную инструкцию создания таблицы и выполняет ее
        self._execute(f'''CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns_with_types)});''')

    def add(self, table_name, data):
        """
        Добавление записи в таблицу БД SQLite
        Args:
            table_name:
            data:
        Returns:
        """
        placeholders = ', '.join('?' * len(data))
        # Ключами являются имена столбцов
        column_names = ', '.join(data.keys())
        # .values() возвращает объект dict_values, но execute требует списка или кортежа
        column_values = tuple(data.values())
        self._execute(f'''INSERT INTO {table_name} ({column_names}) VALUES ({placeholders});''', column_values)

    def delete(self, table_name, criteria):
        """
        Удаление записей из таблицы
        Returns:
        """
        placeholders = [f'{column} = ?' for column in criteria.keys()]
        delete_criteria = ' AND '.join(placeholders)
        self._execute(f'''
        DELETE FROM {table_name}
        WHERE {delete_criteria};
        ''', tuple(criteria.values()),
                      )

    def select(self, table_name, criteria=None, order_by=None):
        """
        будет извлекать все записи
        Args:
            table_name:
            criteria:
            order_by:

        Returns:

        """
        # критерии могут быть пустыми(если в таблице отбираются все записи)
        criteria = criteria or {}

        query = f'SELECT * FROM {table_name}'
        if criteria:
            placeholders = [f'{column} = ?' for column in
                            criteria.keys()]
            select_criteria = ' AND '.join(placeholders)
            query += f' WHERE {select_criteria}'

        # Конструирование предикатного условия ORDER BY для сортировки результатов
        if order_by:
            query += f' ORDER BY {order_by}'
        # возвращает значение из _execute для прокручивания результатов в цикле
        return self._execute(
            query,
            tuple(criteria.values()),
        )

