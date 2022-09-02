"""
@Project ：Engineering_Optimization
@File ：3d_graphs_final.py
@Author ：David Canosa Ybarra
@Date ：31/08/2022 16:36
"""

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

X, Y = np.meshgrid(range(-90, 90, 5), range(1000, 8000, 200))
phases = [i for i in range(0, 390, 30)]
for phase in phases:
    plt.rcParams["figure.figsize"] = 12.8, 9.6

    ax = plt.axes(projection = '3d')

    Z = np.genfromtxt(f"dt100dX25dY2200Z{str(phase)}.csv", delimiter = ",")
    imgname = f"Phase_{str(phase)}"

    ax.plot_wireframe(X, Y, Z)
    ax.plot_surface(X, Y, Z, cmap = 'jet', alpha = 0.5)
    # ax.scatter3D(26.09682, 3382.88681, 1.1419254529722123, color = "purple")
    ###
    ax.set_xlabel("Angle of $\Delta$V [deg]")
    ax.set_ylabel("Magnitude of $\Delta$V [m/s]")
    ax.set_zlabel("Cost of Trajectory [-]")
    ax.view_init(30, (60))
    plt.savefig(f"Graphs\Cost_Surface_{imgname}", bbox_inches = 'tight')
    plt.show()

    # plot filled contour map with 100 levels
    cs = plt.contourf(X, Y, Z, 100, cmap = 'jet')

    # add default colorbar for the map
    cbar = plt.colorbar(cs)

    Z_min_in = np.where(Z == np.min(Z))

    plt.xlabel("Angle of $\Delta$V [deg]")
    plt.ylabel("Magnitude of $\Delta$V [m/s]")
    cbar.set_label("Cost of Trajectory [-]", rotation = 90)
    plt.savefig(f"Graphs\Cost_Contour_{imgname}", bbox_inches = 'tight')
    plt.show()
