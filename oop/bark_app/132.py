import asd

dd = 12
p = asd.Option(io=dd, po=None)

fg = p.strr(dd)
print(fg)
print()

df = p.__str__()
print(df)
