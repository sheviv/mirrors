class Point:
    x = 1
    y = 2
pt = Point()
pt.x = 10
pt.y = 20
print(pt.x)
print(pt.__dict__)  # локальные переменные класса

print(getattr(pt, "x"))  # значение pt.x = 10
print(getattr(pt, "x", False))  # значение pt.x = 10 иначе вернет False
# or
print(hasattr(pt, "y"))

setattr(pt, "z", 7)  # добавить атрибут z=7
delattr(pt, "z")  # удалить любой атрибут

print(isinstance(pt, Point))  #  является ли pt экземпляром классса Point