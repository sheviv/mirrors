# "" - пример
#
# # = sudo

# выводит все разделы, активные в настоящий момент, в удобном для чтения формате.
# df -h

# копируются все файлы и каталоги внутри текущего каталога и создается архивный файл
# tar - создание нового архива, v - гарантирует подробный вывод на экран, f - указывает на имя файла, . - для скрытых файлов вместо *
# tar cvf archivename.tar *
# создает сжатый архив видеофайлов в указанном каталоге.
# tar czvf archivename.tar.gz /home/myuser/Videos/*.mp4
# разделяет большой файл на группу меньших по указанному размеру(-b - разделить файл archivename.tar.gz на части по 1 Гбайт)
# split -b 1G archivename.tar.gz archivename.tar.gz.part
# воссоздать архив, считывая каждую часть по порядку
# cat archivename.tar.gz.part* > archivename.tar.gz

# находит файлы по заданному критерию и передает их имена tar для включения в архив
# -iname возвращает результаты, {} указывают программе fiпd применить команду tar к каждому найденному файлу
# # find /var/www/ -iname "*.mp4" -ехес tar -rvf videos.tar {} \;
# поиск по всей системе файлы, соответствующие указанной строке
# locate *video.mp4
# обновить индексы в ручную
# # updatedb

# удаляет права на чтение для остальных(чтение -r\4, запись -w\2,выполнение -x\1)
# # chmod o-r /bin/zcat
# разрешения на запись для группы
# # chmod g+w /bin/zcat

# создает образ раздела sda2 и сохраняет его в домашнем каталоге
# dd if=/dev/sda2 of=/home/username/partition2.img
# перезаписывает раздел случайными данными, чтобы уничтожить старые данные
# dd if=/dev/ urandom of=/dev/sda1