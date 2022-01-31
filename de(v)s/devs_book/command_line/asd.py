#!/usr/bin/env
import argparse
import sys

# check OS
# print(sys.platform)


# chmode for python.py code
# 1.
# #!/usr/bin/env
# def say():
#     print("Hello, world!")
# if __name__ == '__main__':
#     say()
# 2.
# chmode +x python.py
# 3.
# ./python.py  // Hello, world!


# use sys.argv in command line
#!/usr/bin/env python3
# if __name__ == '__main__':
#     print(f"first: {sys.argv[0]}")
#     print(f"second: {sys.argv[1]}")
# ./asd.py qwe zxc  // first: ./asd.py \n second: qwe


# argparse
# 1.
#!/usr/bin/env python3
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="echo:")
#     parser.add_argument("message", help=",sg echo")
#     parser.add_argument('--twice', '-t', help='do twice', action='store_true')
#     args = parser.parse_args()
#     if args.twice:
#         print(f"{args.message}")
# 2.
# python3 asd.py --twice my_message  // my_message


# simple decorators
# def aaa(twice_func):
#     def twice():
#         print("1")
#         twice_func()
#         print("2")
#     return twice
# def bbb():
#     print("bbb")
# f = aaa(bbb)
# f()
# or if use decorator
# @aaa
# def new_bbb():
#     print("new_bbb")
# new_bbb()


"""
Click
"""
#!/usr/bin/env python3
import click
# @click.command()
# @click.option('--greeting', default="Hiya", help="How are you?")
# @click.option('--name', default="Tammy", help="How are you?")
# def you(greeting, name):
#     print(f"{greeting}, {name}")
# if __name__ == "__main__":
#     you()


"""
Fire
"""
# 1.
#!/usr/bin/env python
# import fire
# def hello(name):
#     return 'Hello {name}!'.format(name=name)
# def goodbye(second):
#     return 'Goodbye {name}!'.format(name=second)
# if __name__ == '__main__':
#     fire.Fire()
# 2.
#  python3 asd.py hello World


"""
Numba
"""
# import time
# import numba
# def timing(f):
#     @wraps(f)
#     def wrap(*args, **kwargs):
#         ts = time()
#         result = f(*args, **kwargs)
#         te = time()
#         print(f"fun: {f.__name__}, args: [{args}, {kwargs}] took: {te-ts} sec")
#         return result
#     return wrap
#
# @timing
# @numba.jit(nopython=True)
# def expmean_jit(rea):
#     val = rea.mean() ** 2
#     return val
