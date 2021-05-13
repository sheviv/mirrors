# Создание таблицы, добавление данных, проверка и вывод.

import sqlite3 as sq

# create db
db = sq.connect("server.db")
# change in db
sql = db.cursor()

# create table if not exists
sql.execute("""CREATE TABLE IF NOT EXISTS users (
login TEXT,
password TEXT,
cash BIGINT
)""")

# confirmed anything
db.commit()

# input login and password
user_login = input("Login: ")
user_password = input("Password: ")

# select column login in table users
sql.execute("SELECT login FROM users")
# add value in table
if sql.fetchone() is None:
    sql.execute(f"INSERT INTO users VALUES (?, ?, ?)", (user_login, user_password, 0))
    db.commit()
else:
    print("Запись уже есть.")
    # print values in table
    for value in sql.execute("SELECT * FROM users"):
        print(value[0])
