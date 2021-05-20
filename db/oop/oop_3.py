class Point:
    def __init__(self, x=0, y=0):  # конструктор при вызове экзепляра класса
        # self.x = x
        # self.y = y
        # приватные атрибуты __name
        self.__x = x
        self.__y = y
    def setCoords(self,x ,y):
        self.__x = x
        self.__y = y

    def getCoords(self):
        return self.__x, self.__y

pt = Point(2,3)
print(pt.getCoords())
# pt.setCoords(4,5)
# print(pt.getCoords())

print(pt._Point__y)  # обращение к атрибуту y клааса Point


# перегруженные методы классов
# youtube/selfedu/ООП#3
# __settar - вызывается при изменении св-ва лкасса
# __getattribute__ - вызывается при получении св-ва класса с именем  item
# __getattr__ - вызывается при получении несуществующего св-ва item класса
# __delattr__ - вызывается при удалении св-ва item


# Объекты св-ва property
class Point:
    def __init__(self, x=0, y=0):
        # self.x = x
        # self.y = y
        # приватные атрибуты __name
        self.__x = x
        self.__y = y
    def __getCoords(self):
        print("2")
        return self.__x

    def __setCoords(self,x):
        print("1")
        self.__x = x

    coordZ = property(__getCoords, __setCoords)
pt = Point(2,3)
pt.coordZ = 100  # __setCoords запись значения
x = pt.coordZ  # __getCoords чтение значения
# print(pt.getCoords())