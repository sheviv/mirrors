# "" - пример
#
# # = sudo

# Веб-серверы: создание сервера MediaWiki

# единственная команда Ubuntu, устанавливающая все элементы сервера LAMP
# apt install lamp- server^
# запускает Apache на CentOS при каждой загрузке системы
# systemctl enable httpd
# разрешает трафик НТТР-браузера в системе CentOS
# firewall -cmd --add -service=http--permanent
# сбрасывает ваш административный пароль и повышает безопасность базы данных
# mysql_secure_installation
# позволяет войти в MySQL(илиMariaDB) в качестве rооt-пользователя
# mysql -u root -р
# создает новую базу данных в MySQL (или MariaDB)
# CREATE DATABASE newdbname;
# ищет доступные пакеты, связанные с РНР, на компьютере с CentOS
# yum search php- | grep mysql
# выполняет поиск доступных пакетов, связанных с многобайтовым кодированием строк
# apt search mbstring
