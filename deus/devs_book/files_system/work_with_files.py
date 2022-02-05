# Чтение и запись файлов
# file_path = 'test.txt'
# open_file = open(file_path, 'r')
# text = open_file.read()
# print(len(text))
# open_file.close()

# разбивает его содержимое по символам перевода строк.
# open_file = open(file_path, 'r')
# text = open_file.readlines()
# print(text[0])
# open_file.close()

# Python сам закроет файл
# with open(file_path, 'r') as open_file:
#     text = open_file.readlines()
#     print(text[1])

# для чтения двоичных файлов можно добавить b в модификатор режима
# file_path = 'bookofdreamsghos00lang.pdf'
# with open(file_path, 'rb') as open_file:
#     btext = open_file.read()


"""
JSON
"""
# прочитать текст из файла json
import pathlib
# with open('policy.json', 'r') as opened_file:
#     policy = opened_file.readlines()
#     print(policy)
# or
import json
# with open('policy.json', 'r') as opened_file:
#     policy = json.load(opened_file)
    # print(policy)
# for full view
from pprint import pprint
# pprint(policy)

# add new value in json file
# policy['Statement']['Resource'] = 'S3'
# pprint(policy)


"""
YAML
"""
# pip install PyYAML
import yaml
# with open('verify.yml', 'r') as opened_file:
#     verify_apache = yaml.safe_load(opened_file)
#     pprint(verify_apache)

# сохранять данные из Python в формате YAML
# with open('verify-apache.yml', 'w') as opened_file:
#     yaml.dump(verify_apache, opened_file)


"""
YAML
"""
import xml.etree.ElementTree as ET
# tree = ET.parse('http_feeds.feedburner.com_oreilly_radar_atom.xml')
# root = tree.getroot()
# for child in root:
#     print(child.tag, child.attrib)


"""
CSV
"""
import csv
# file_path = 'addresses.csv'
# with open(file_path, newline='') as csv_file:
#     off_reader = csv.reader(csv_file, delimiter=',')
#     for _ in range(5):
#         print(next(off_reader))


"""
Change CSV to JSON
"""
contents = open('name.csv').readlines()
json.dumps(list(csv.reader(contents)))


"""
Pandas
"""
import pandas as pd
# df = pd.read_csv('username.csv', sep=";")
# print(df.head(3))
# print(df["Username"])


"""
re
"""
import re
# line = '127.0.0.1 - rj [13/Nov/2019:14:43:30 -0000] "GET HTTP/1.0" 200'
# m = re.search(r'(?P<IP>\d+\.\d+\.\d+\.\d+)', line)
# m.group('IP')


# Построчное чтение и запись в новый файл
# with open('big-data.txt', 'r') as source_file:
#     with open('correct-data.txt', 'w') as target_file:
#         for line in source_file:
#             target_file.write(line)


"""
Шифрование текста
"""
# SHA1, SHA224, SHA384, SHA512, MD5
# Через библиотеку hashlib
import hashlib
# secret = "This is example for my fisrt encripting."
# b_cesret = secret.encode()
# m = hashlib.md5()
# m.update(b_cesret)
# print(m.digest())

# Через библиотеку cryptography
from cryptography.fernet import Fernet
# key = Fernet.generate_key()  # key for encripting
# encripting data(text)
# f = Fernet(key)
# msg = b"My secret is very simple."
# encrypt = f.encrypt(msg)
# decrypting data
# f = Fernet(key)
# df = f.decrypt(encrypt)

# Double(key-public, key-private)
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
# privat_key = rsa.generate_private_key(public_exponent=65537,
#                                       key_size=4096,
#                                       backend=default_backend())
# public_key = privat_key.public_key()
# ecrypting data with public key
# msh = b"More my secrets."
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
# encrypted = public_key.encrypt(msh,
#                                padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
#                                             algorithm=hashes.SHA256(),
#                                             label=None))
# decrypting data with private key
# decrypted = privat_key.decrypt(encrypted,
#                                padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
#                                             algorithm=hashes.SHA256(),
#                                             label=None))
# print(decrypted)  # More my secrets.
