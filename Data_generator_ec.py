"""
@Project ：Engineering_Optimization
@File ：Data_generator.py
@Author ：David Canosa Ybarra
@Date ：30/08/2022 21:08
"""
from Simulator import *

from time import *
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

X2, Y2 = np.meshgrid(range(-90, 90, 1), range(1000, 8000, 40))


def Z2_fun(angle, magnitude):
    Z2 = np.zeros(np.shape(angle))
    for i in range(len(angle)):

        for j in range(len(angle[0])):
            print("Current Progress")
            ShowLoading(i, len(angle), len(angle))
            ShowLoading(j, len(angle[0]), len(angle[0]))
            start = time()
            print("\n")
            norm_dv = np.array([[np.cos(np.deg2rad(angle[i][j]))],
                                [np.sin(np.deg2rad(angle[i][j]))],
                                [0]])
            dv_man = norm_dv * magnitude[i][j]
            x, x_moon = Simulate(SimplePropagator, 0, 120, 400000, dv_man, dt = 100)
            Z2[i][j] = Cost3(x, x_moon, dv_man, 1, 1)
            end = time()
            time_taken = end - start
            time_left_sec = (len(angle) * len(angle[0]) - ((i) * len(angle[0]) + (j))) * time_taken
            time_left_hr = time_left_sec // (60 * 60)
            time_left_min = (time_left_sec // (60)) - (60 * time_left_hr)
            print(f"{int(time_left_hr)}:{int(time_left_min)}\tTime left until completion.")
    return Z2


Z2 = Z2_fun(X2, Y2)

np.savetxt("dt100dX21dY240_ec.csv", Z2, delimiter = ",")
# # https://stackoverflow.com/questions/51860716/how-save-a-array-to-text-file-in-python
# with open(f"output_{ctime()}.txt", "w") as txt_file:
#     for line in Z2:
#         txt_file.write(" ".join(line) + "\n") # works with any number of elements in a line
#
# ###
#
# ax = plt.axes(projection = '3d')
# ax.plot_surface(X2, Y2, Z2, cmap = 'jet')
# plt.show()
