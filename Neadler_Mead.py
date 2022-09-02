"""
@Project ：Engineering_Optimization
@File ：Neadler_Mead.py
@Author ：David Canosa Ybarra
@Date ：01/09/2022 18:30
"""
from Simulator import *

from time import *
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def Optimize_Function(x):
    angle = x[0]
    magnitude = x[1]

    norm_dv = np.array([[np.cos(np.deg2rad(angle))],
                        [np.sin(np.deg2rad(angle))],
                        [0]])
    dv_man = norm_dv * magnitude
    x, x_moon = Simulate(SimplePropagator, 0, 120, 400000, dv_man, dt = 100)
    cost = Cost(x, x_moon, dv_man, 1, 1)
    return cost


x0 = np.array([20, 6000])
res = minimize(Optimize_Function, x0, method = 'nelder-mead', options = {'xatol': 1e-8, 'disp': True})

raise IndexError

plt.rcParams["figure.figsize"] = 12.8, 9.6

ax = plt.axes(projection = '3d')

X, Y = np.meshgrid(range(-90, 90, 5), range(1000, 8000, 200))

Z = np.genfromtxt("Z2.csv", delimiter = ",")
imgname = "_Optimized"

ax.plot_wireframe(X, Y, Z)
ax.plot_surface(X, Y, Z, cmap = 'jet', alpha = 0.5)
ax.scatter3D(res.x[0], res.x[1], Optimize_Function(x), c = 'r')
###
ax.set_xlabel("Angle of $\Delta$V [deg]")
ax.set_ylabel("Magnitude of $\Delta$V [m/s]")
ax.set_zlabel("Cost of Trajectory [-]")
ax.view_init(30, (60))
plt.savefig(f"Cost_Surface_{imgname}", bbox_inches = 'tight')
plt.show()

# plot filled contour map with 100 levels
cs = plt.contourf(X, Y, Z, 100, cmap = 'jet')

# add default colorbar for the map
cbar = plt.colorbar(cs)
plt.xlabel("Angle of $\Delta$V [deg]")
plt.ylabel("Magnitude of $\Delta$V [m/s]")
cbar.set_label("Cost of Trajectory [-]", rotation = 90)
plt.savefig(f"Cost_Contour{imgname}", bbox_inches = 'tight')
plt.show()
