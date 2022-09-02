"""
@Project ：Engineering_Optimization
@File ：3d_graphs_final.py
@Author ：David Canosa Ybarra
@Date ：31/08/2022 16:36
"""
from Simulator import *

from time import *
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = 12.8, 9.6

ax = plt.axes(projection = '3d')

X, Y = np.meshgrid(range(-90, 90, 1), range(1000, 8000, 40))

Z = np.genfromtxt("Data/dt100dX21dY240_ec.csv", delimiter = ",")
imgname = "_High_Res_EC"

ax.plot_wireframe(X, Y, Z)
ax.plot_surface(X, Y, Z, cmap = 'jet', alpha = 0.5)
# ax.scatter3D(26.09682, 3382.88681, 1.1419254529722123, color = "purple")
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
# Optimization terminated successfully.
#          Current function value: 1.141925
#          Iterations: 92
#          Function evaluations: 203
# (60, 6000)= 26.09682, 3382.88681
# Optimization terminated successfully.
# Current function value: 1.309558
# Iterations: 84
# Function evaluations: 168
# (60, 6000)= 21.89719394321709, 3928.6754761701295

Z_min_in = np.where(Z == np.min(Z))

# plt.plot(X[Z_min_in], Y[Z_min_in], color = "red", marker = "X", markersize=15)
plt.plot(21.89719394321709, 3928.6754761701295, color = "white", marker = "X", markersize = 15)
plt.xlabel("Angle of $\Delta$V [deg]")
plt.ylabel("Magnitude of $\Delta$V [m/s]")
cbar.set_label("Cost of Trajectory [-]", rotation = 90)
plt.savefig(f"Cost_Contour{imgname}", bbox_inches = 'tight')
plt.show()
