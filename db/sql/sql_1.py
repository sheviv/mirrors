import sqlite3 as sq

# откроет дб, если ее нет - создаст
with sq.connect("saper.db") as con:
    cur = con.cursor()  # Cursor для работы с бд

    # удалить таблицу
    cur.execute("DROP TABLE users")

    # создание таблицы с названием столбца и его типом
    # IF NOT EXISTS для избежания ошибок при повторном запуске
    cur.execute("""CREATE TABLE IF NOT EXISTS users (
    name TEXT, 
    sex INTEGER NOT NULL DEFAULT 1,
    old INTEGER,
    score INTEGER
    )""")
    # user_id INTEGER PRIMARY KEY - должно содержать уникальные значения
    # AUTOINCREMENT - значение поля user_id должно увеличиваться на 1(для PRIMARY KEY)
    # NOT NULL DEFAULT 1 - если нет значения - по дефолту 1

# ДОБАВИТЬ ЗНАЧЕНИЕ В ТАБЛИЦУ
# INSERT INTO users VALUES('Mihail', 2, 17, 2000)
# ДОБАВИТЬ ЗНАЧЕНИЯ В name, old, score
# INSERT INTO users (name, old, score) VALUES('Fyodor', 7, 1000)

# ПОКАЗАТЬ ВСЕ ПОЛЯ ИЗ ТАБЛИЦЫ
# SELECT * FROM users
# ПОКАЗАТЬ ПОЛЯ name, old, score ИЗ ТАБЛИЦЫ
# SELECT name, old, score FROM users
# ПОКАЗАТЬ ЗНАЧЕНИЯ(С УСЛОВИЕМ), ГДЕ score < 1500
# SELECT * FROM users WHERE score < 1500
# ПОКАЗАТЬ ЗНАЧЕНИЯ(С УСЛОВИЕМ), В ПРОМЕЖУТКЕ 500 - 1500
# SELECT * FROM users WHERE score BETWEEN 500 AND 1500
# ПОКАЗАТЬ ЗНАЧЕНИЯ(С УСЛОВИЕМ), ГДЕ score == 2000
# SELECT * FROM users WHERE score == 2000

# СОРТИРОВАТЬ ПО УВЕЛИЧЕНИЮ(old ASC) ЗНАЧЕНИЙ В СТОЛБЦЕ old (УМЕНЬШЕНИИ old DESC)
# ORDER BY old ASC
# ОГРАНИЧЕНИЕ ВЫБОРКИ(ПОКАЗАТЬ M ЗАПИСЕЙ, ПОСЛЕ N ЗНАЧЕНИЙ - ОТСТУП)
# LIMIT N,M

# ОБНОВИТЬ ДАННЫЕ для 1 значения
# UPDATE users SET score = 1230 WHERE rowid = 1
# ОБНОВИТЬ ДАННЫЕ для значения именем Fyodor
# UPDATE users SET score = 1230 WHERE name LIKE 'Fyodor'
# ОБНОВИТЬ ДАННЫЕ для значения именем НАЧИНАЮЩИМСЯ С F%
# % - продолжение строки
# UPDATE users SET score = 1230 WHERE name LIKE 'F%'
# ОБНОВИТЬ ДАННЫЕ для значения именем НАЧИНАЮЩИМСЯ С F_od%
# _ - любой символ
# UPDATE users SET score = 1230 WHERE name LIKE 'F_od%'

# УДАЛИТЬ ЗНАЧЕНИЯ rowid ИЗ ТАБЛИЦЫ СО ЗНАЧЕНИЯМИ 2,5
# DELETE FROM users WHERE rowid IN(2, 5)

# ГРУППИРОВКА ЗНАЧЕНИЙ GROUP BY(ПОДСЧЕТ СУММЫ sum ДЛЯ КАЖДОГО user_id ИЗ ТАБЛИЦЫ games ПО УБЫВАНИЮ DESC)
# SELECT user_id, sum(score) AS sum
# FROM games
# GROUP BY user_id
# ORDER BY sum DESC

# ОБЪЕДИНЕНИЕ НЕСКОЛЬКИХ ТАБЛИЦ UNION(где score=val, 'from'=type)
# UNION - ТОЛЬКО УНИКАЛЬНЫЕ ЗНАЧЕНИЯ
# SELECT score, `from` FROM tab1
# UNION SELECT val, type FROM tab2

# ХРАНЕНИЕ ГРАФИЧЕСКИХ ДАННЫХ И СОЗДАНИЕ КОПИИ БД
# https://www.youtube.com/watch?v=Ic6etzJZF-M&list=PLA0M1Bcd0w8x4Inr5oYttMK6J47vxgv6J&index=10&ab_channel=selfedu