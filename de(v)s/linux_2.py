# "" - пример
#
# # = sudo
#
# Работа с менеджерами пакетов Linux
# установить VirtualBox на машине
# # apt update
# # apt install virtualbox
#
# установить файл из оболочки командной строки(-i(для установки))
# dpkg позаботится обо всех программных зависимостях
# cd /home/"username"/Downloads
# # dpkg -i skypeforlinux-64.deb
# структура каталогов /etc/
# информация о репозитории хранится по адресу /etc/yum.repos.d/
# wget - для загрузки файла .repo
# cd /etc/yum.repos.d/
# # wget http://download.virtualbox.org/virtualbox/rpm/fedora/virtualbox.repo
# обновление индексов и файлов
# # dnf update
#
# сокращенный вариант процесса установки зависимостей
# # dnf install patch kernel-devel dkms
# установка нуджной версии
# # dnf install VirtualBox-"5.1"
#
# Если дополнения vbox еще недоступны на хосте, установить пакет расширений в Ubuntu
# sudo apt install virtualbox-guest-additions-iso

# поиск пакетов
# apt search sensors
# отображение полной информации о пакете
# apt show lm-sensors

# получить хэш сумму загруженного файла
# shasum ubuntu-16.04.2-server-amd64.iso

# отображение всех виртуальных машин, доступных в настоящее время в системе
# vboxmanage list vms

# клонирование для графического интерфейса
# vboxmanage clonevm --register "Kali"-Linux-template --name newkali

# экспортировать локальную виртуальную машину в файл, используя открытый формат виртуализации
# флаг -o указывает выходное имн файла
# vboxmanage export website-project -о "website.ova"

# копирование через сеть
# scp website.ova username@192.168.0.34: "/home/username"

# импортировать с удаленного компьютера виртуальную машину в VirtualBox
# vboxmanage import website.ova

# успешно ли операция импорта завершена
# vboxmanage list vms


# Работа с контейнерами Linux (LXC)
# установка LXC на рабочую станцию Ubuntu
# # apt update
# # apt install lxc

# Создание контейнера
# -n имя контейнера, -t указывает LXC построить контейнер по шаблону Ubuntu
# # lxc -create -n myContainer -t ubuntu

# изменить пароль
# passwd
# проверка состояния контейнера
# # lxc -ls --fancy

# -d  «отсоединение»(то есть не будет автоматического запуска интерактивной сессию при запуске контейнера)
# без -d единственный способ выйти из сессии - закрыть контейнер
# # lxc -start -d -n myContainer
# Список контейнеров
# # lxc -ls --fancy

# запустить сеанс оболочки администратора в работающем контейнере
# # lxc-attach -n myContainer

# выйти, оставив контейнер работающим
# exit
# завершить работу контейнера(-h это остановка)
# shutdown -h now
# перезагрузить контейнер(-r reboot)
# shutdown -r now
