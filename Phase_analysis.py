"""
@Project ：Engineering_Optimization
@File ：Phase_analysis.py
@Author ：David Canosa Ybarra
@Date ：02/09/2022 01:06
"""
from Simulator import *

from time import *
import numpy as np

X2, Y2 = np.meshgrid(range(-90, 90, 5), range(1000, 8000, 200))
phases = [i for i in range(0, 390, 30)]
for phase in phases:
    def Z2_fun(angle, magnitude):
        Z2 = np.zeros(np.shape(angle))
        EC = np.zeros(np.shape(angle))
        for i in range(len(angle)):

            for j in range(len(angle[0])):
                print("Current Progress")
                ShowLoading(phases.index(phase), len(phases), len(phases))
                ShowLoading(i, len(angle), len(angle))
                ShowLoading(j, len(angle[0]), len(angle[0]))
                start = time()
                print("\n")
                norm_dv = np.array([[np.cos(np.deg2rad(angle[i][j]))],
                                    [np.sin(np.deg2rad(angle[i][j]))],
                                    [0]])
                dv_man = norm_dv * magnitude[i][j]
                x, x_moon = Simulate(SimplePropagator, 0, phase, 400000, dv_man, dt = 100)
                Z2[i][j], EC[i][j] = Cost2(x, x_moon, dv_man, 1, 1)
                end = time()
                time_taken = end - start
                time_left_sec = (len(angle) * len(angle[0]) - ((i) * len(angle[0]) + (j))) * time_taken * (
                        len(phases) - phases.index(phase))
                time_left_hr = time_left_sec // (60 * 60)
                time_left_min = (time_left_sec // (60)) - (60 * time_left_hr)
                print(f"{int(time_left_hr)}:{int(time_left_min)}\tTime left until completion.")
        return Z2, EC


    Z2, EC = Z2_fun(X2, Y2)

    np.savetxt(f"Phase_analysis\dt100dX25dY2200Z{str(phase)}.csv", Z2, delimiter = ",")

    np.savetxt(f"Phase_analysis\dt100dX25dY2200EC{str(phase)}.csv", EC, delimiter = ",")
