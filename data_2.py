"""
@Project ：Engineering_Optimization
@File ：data.py
@Author ：David Canosa Ybarra
@Date ：31/08/2022 14:37
"""
from Simulator import *

from time import *
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

X2, Y2 = np.meshgrid(range(-90, 90, 20), range(1000, 8000, 800))
Z2 = np.array([[72.55035, 72.35471, 72.11900, 71.90025, 71.76949, 71.77591, 71.91838, 72.14637, 72.38881],
               [72.61643, 72.15538, 71.49428, 70.74786, 70.21929, 70.23810, 70.79283, 71.54754, 72.20697],
               [72.62038, 71.69299, 69.95248, 67.03495, 63.86126, 63.95788, 67.21082, 70.08343, 71.77910],
               [72.52681, 70.65820, 64.61642, 7.30052, 28.44079, 13.37488, 11.10546, 65.18869, 70.82757],
               [72.26504, 67.94357, 66.85803, 36.13596, 13.86833, 4.33700, 14.10255, 18.41215, 68.40437],
               [71.67616, 54.42361, 55.12215, 25.94691, 4.49494, 14.50486, 25.66879, 31.85998, 57.77574],
               [70.33697, 73.22297, 47.27761, 19.23976, 5.54453, 21.39738, 33.34626, 41.51143, 43.59992],
               [66.60520, 71.68098, 41.77231, 14.57056, 10.42658, 26.39433, 38.82741, 48.05404, 53.25147],
               [47.43025, 68.81120, 37.75157, 11.19599, 14.12400, 30.19713, 42.95634, 52.78672, 59.47666]])

cs = plt.contour(Z2, extend = 'both')
cs.cmap.set_over('red')
cs.cmap.set_under('blue')
cs.changed()
plt.show()