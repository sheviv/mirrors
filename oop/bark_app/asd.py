class Option:
    def __init__(self, name):
        self.name = name
        # self.io = io

    def __str__(self):
        return self.name

    def strr(self, io):
        return f"{io}"


df = [123, "qwe", "45"]
for i in df:
    cv = Option(i)
    dd = cv.__str__()
    print(type(dd))