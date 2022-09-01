"""
@Project ：Engineering_Optimization
@File ：Test.py
@Author ：David Canosa Ybarra
@Date ：29/08/2022 18:33
"""
from sympy import *

x = Symbol('x')
expr = integrate(x + x ** x, x)
