# "" - пример
#
# # = sudo

# Инструменты для критических ситуаций: создание устройства для восстановления системы

# вычисляет контрольную сумму SHA256 файла в формате ISO
# sha256sum systemrescuecd-x86-5.0.2.iso
# добавляет запись MBR с поддержкой USВ-носителей к образу загрузочного диска
# isohybrid systemrescuecd-x86-5.0.2.iso
# записывает образ загрузочного диска на пустой носитель
# dd bs=4M if=systemrescuecd-x86-5.0.2.iso of=/dev/sdb && sync
# монтирует раздел в каталог в загрузочной файловой системе
# mount /dev/sdc1/run/temp-directory
# сохраняет файлы из поврежденного раздела в образ с именем sdc1-backup.img и записывает события в файл журнала
# dd rescue -d /dev/sdc1/run/usb-mount/sdc1-backup.img/run/usb-mount/sdc1-backup.logfile
# открывает оболочку от имени администратора в файловой системе
# chroot /run/mountdir/
