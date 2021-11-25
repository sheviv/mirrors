import sys
from datetime import datetime

from database import DatabaseManager

# создаст файл БД, если тот не существует
db = DatabaseManager('bookmarks.db')
class CreateBookmarksTableCommand:
    # Будет вызвано при запуске приложения Barks
    def execute(self):
        """
        Создает таблицу bookmarks с необходимыми столбцами и ограничениями
        Returns:
        """
        db.create_table('bookmarks', {
            'id': 'integer primary key autoincrement',
            'title': 'text not null',
            'url': 'text not null',
            'notes': 'text',
            'date_added': 'text not null',
        })




# AddBookmarkCommand для выполнения этой операции. Он будет:
# 1) ожидать словарь, содержащий заголовок, URL-адрес и, возможно, примечание;
# 2) добавлять текущую дату и время в словарь в date_added. Чтобы получить текущее время в стандартизированном формате UTC,
# используйте datetime.datetime.utcnow().isoformat();
# 3) вставлять данные в таблицу bookmarks методом DatabaseManager.add;
# 4) возвращать сообщение об успехе, которое потом будет показываться в слое визуализации.
from datetime import datetime
class AddBookmarkCommand:
    def execute(self, data):
        # Добавляет текущую дату и время при добавлении записи
        data['date_added'] = datetime.utcnow().isoformat()
        # Использование метода .add класса DatabaseManager укорачивает добавление записи
        db.add('bookmarks', data)
        return 'Закладка добавлена!'



class ListBookmarksCommand:
    """
    Класс для вывода на экран списка существующих закладок
    """
    def __init__(self, order_by='date_added'):
        # можно создать версию этой команды для сортировки по дате и по заголовку
        self.order_by = order_by
    def execute(self):
        """
        возвращает курсор, который можно прокрутить в цикле для получения записей
        Returns:
        """
        return db.select('bookmarks', order_by=self.order_by).fetchall()


#
class DeleteBookmarkCommand:
    """
    Класс для удаления закладок
    """
    def execute(self, data):
        """
        принимает словарь имен столбцов и сопоставляет пары значений
        Args:
            data:
        Returns:
        """
        db.delete('bookmarks', {'id': data})
        return 'Bookmark deleted!'


class QuitCommand:
    """
    Класс для выхода из программы
    """
    def execute(self):
        sys.exit()