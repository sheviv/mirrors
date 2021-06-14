class Point:
    def __init__(self, x, y):  # конструктор при вызове экзепляра класса
        self.x = x
        self.y = y
    def __del__(self):  # деструктор для удаления экземпляра
        """
        удаляется, если у него нет экземпляров класса
        """
        pass
    def setCoords(self, x, y):
        self.x = x
        self.y = y
        return self.x, self.y
pt = Point(3,4)
print(pt.setCoords(2, 3))