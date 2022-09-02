"""
@Project ：Engineering_Optimization
@File ：Simulator.py
@Author ：David Canosa Ybarra
@Date ：28/06/2022 15:43

Version 1.0.1

General things to do:
TODO: Implement a way to visualize the data [DONE]
TODO: Add better propagators
TODO: Add performance metric
"""
import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt

G, M_earth, M_moon, alt_moon = 6.674 * 10 ** -11, 5.972 * 10 ** 24, 7.342 * 10 ** 22, 3.844 * 10 ** 8
R_earth, ISS_alt, R_moon = 6.378 * 10 ** 6, 4.000 * 10 ** 5, 1.738 * 10 ** 6


def Simulate(propagator, alpha, beta, alt, dv, dt = 15, G = G, M_earth = M_earth, M_moon = M_moon):
    """
    :param propagator: Numerical state propagator
    :param alpha: Starting phase of spacecraft
    :param beta: Starting phase of the moon
    :param alt: Spacecraft starting altitude
    :param dv: Manoeuvre delta-v vector
    :param dt: Time-step
    :param G: Constant
    :param M_earth: Constant
    :param M_moon: Constant
    :return: Trajectory positions
    """
    # 1. Set initial positions and velocities
    x_earth = np.zeros((3, 1))
    x_moon = MoonPosition(beta, 0)
    x, x0_dot = InitialConditions(alpha, alt)
    x_dot = x0_dot + dv
    x_dot_dot = np.zeros((3, 1))
    # 2. Initialise time array and set an ending time for the simulation
    t = np.array([0])
    t_end = 10 * (24 * 60 * 60)
    # 3. Begin simulation
    while t[-1] < t_end:
        # 3a. Set current moon position
        x_moon = np.hstack((x_moon, MoonPosition(beta, t[-1])))  # FIXME: Has n+1 list
        # 3b. Solve the equations of motion to find the accelerations at time t
        x_dot_dot_earth = (x_earth[:, -1:] - x[:, -1:]) * ((G * M_earth) / la.norm(x_earth[:, -1:] - x[:, -1:]) ** 3)
        x_dot_dot_moon = (x_moon[:, -1:] - x[:, -1:]) * ((G * M_moon) / la.norm(x_moon[:, -1:] - x[:, -1:]) ** 3)
        x_dot_dot_total = x_dot_dot_earth + x_dot_dot_moon

        x_dot_dot = np.hstack((x_dot_dot, x_dot_dot_total))
        # 3c. Use the propagator to find the position at time t + dt
        x_dot = np.hstack((x_dot, propagator(x_dot_dot[:, -1:], x_dot[:, -1:], dt)))
        x = np.hstack((x, propagator(x_dot[:, -1:], x[:, -1:], dt)))
        # 3d. Set the new time for the loop to start at and print progress bar
        t = np.hstack((t, t[-1] + dt))
        # ShowLoading(t[-1], t_end)
    # 4. End the simulation and return the position values for the spacecraft and the moon at each respective time
    return x, x_moon


def Cost(x, x_moon, dv, c1, c2, R_moon = R_moon):
    # Check for moon intercept
    dist_m = np.array([])
    for i in range(len((x - x_moon)[1, :])):
        dist_m = np.hstack((dist_m, [la.norm((x - x_moon)[:, i])]))
    # Check for earth collision prob remove for optimizer
    # dist_e = np.array([])
    # for i in range(len((x)[1, :])):
    #     dist_e = np.hstack((dist_e, [la.norm((x)[:, i])]))
    # ec = min(dist_e)
    #
    # if ec <= (R_earth + 100000):
    #     Earth_Collision = np.nan
    # else:
    #     Earth_Collision = 0
    a = min(dist_m)  # + Earth_Collision
    b = la.norm(dv)
    norm_a = a / (3 * R_moon)
    norm_b = b / 3000
    cost = c1 * norm_a + c2 * norm_b
    return cost


def Cost2(x, x_moon, dv, c1, c2, R_moon = R_moon):
    # Check for moon intercept
    dist_m = np.array([])
    for i in range(len((x - x_moon)[1, :])):
        dist_m = np.hstack((dist_m, [la.norm((x - x_moon)[:, i])]))
    # Check for earth collision prob remove for optimizer
    dist_e = np.array([])
    for i in range(len((x)[1, :])):
        dist_e = np.hstack((dist_e, [la.norm((x)[:, i])]))
    ec = min(dist_e)
    #
    # if ec <= (R_earth + 100000):
    #     Earth_Collision = 1
    # else:
    #     Earth_Collision = 0
    a = min(dist_m)
    b = la.norm(dv)
    norm_a = a / (3 * R_moon)
    norm_b = b / 3000
    cost = c1 * norm_a + c2 * norm_b
    return cost, ec


def Cost3(x, x_moon, dv, c1, c2, R_moon = R_moon):
    # Check for moon intercept
    dist_m = np.array([])
    for i in range(len((x - x_moon)[1, :])):
        dist_m = np.hstack((dist_m, [la.norm((x - x_moon)[:, i])]))
    # Check for earth collision prob remove for optimizer
    dist_e = np.array([])
    for i in range(len((x)[1, :])):
        dist_e = np.hstack((dist_e, [la.norm((x)[:, i])]))
    ec = min(dist_e)

    if ec <= (R_earth + 100000):
        Earth_Collision = np.nan
    else:
        Earth_Collision = 0
    a = min(dist_m) + Earth_Collision
    b = la.norm(dv)
    norm_a = a / (3 * R_moon)
    norm_b = b / 3000
    cost = c1 * norm_a + c2 * norm_b
    return cost


def SimplePropagator(x_dot, x, dt):
    """
    A simple 1st order propagator using the euler method.
    :param x_dot: Function gradient
    :param x: Function at point t
    :param dt: Time-step
    :return: Function at point t+dt
    """
    xplusdt = x + x_dot * dt
    return xplusdt


def InitialConditions(alpha, alt, G = G, M_earth = M_earth, R_earth = R_earth):
    """
    :param alpha: Initial phase of the spacecraft
    :param alt: Starting altitude of the spacecraft
    :param G: Constant
    :param M_earth: Constant
    :param R_earth: Constant
    :return: Starting x and x_dot
    """
    x = np.array([[(alt + R_earth) * np.sin(np.deg2rad(alpha))],
                  [(alt + R_earth) * np.cos(np.deg2rad(alpha))],
                  [0]])
    V = (G * M_earth / (alt + R_earth)) ** 0.5
    x_dot = np.array([[V * np.cos(np.deg2rad(alpha))],
                      [V * -np.sin(np.deg2rad(alpha))],
                      [0]])
    return x, x_dot


def MoonPosition(beta, t, alt_moon = alt_moon, M_earth = M_earth, G = G):
    """
    :param beta: Initial phase of the moon
    :param t: Current time elapsed
    :param alt_moon: Distance from the center of the moon to the center of earth
    :param M_earth: Constant
    :param G: Constant
    :return: Current moon position
    """
    T = 2 * np.pi * (alt_moon ** 3 / (G * M_earth)) ** 0.5
    beta_now = np.rad2deg(((t % T) / T) * 2 * np.pi)
    x_moon = np.array([[alt_moon * np.sin(np.deg2rad(beta + beta_now))],
                       [alt_moon * np.cos(np.deg2rad(beta + beta_now))],
                       [0]])
    return x_moon


def ShowLoading(t, t_end, length = int(20)):
    Loaded = int((t / t_end) * length) * "|"
    Todo = (length - int((t / t_end) * length)) * ":"
    print(f"[{Loaded}{Todo}]")


def circle(r, p = (0, 0)):
    theta = np.deg2rad(np.arange(0, 360, 1))
    x = r * np.sin(theta) + p[0]
    y = r * np.cos(theta) + p[1]
    return x, y


def display(x, x_moon, R_earth = R_earth, R_moon = R_moon):
    fig, axs = plt.subplots(1, 1, figsize = (5, 5), dpi = 400)

    x_c, y_c = circle(R_earth)
    axs.plot(x_c, y_c, color = "blue", linewidth = 1.5)
    x_c_m, y_c_m = circle(R_moon, (x_moon[0, -1], x_moon[1, -1]))

    axs.plot(x_moon[0, :], x_moon[1, :], color = "grey", linewidth = 0.5, linestyle = "dashed")
    axs.plot(x_c_m, y_c_m, color = "grey", linewidth = 1.5)

    axs.plot(x[0, :], x[1, :], color = "red", linewidth = 1, linestyle = "dashed")

    dist = np.array([])
    for i in range(len((x - x_moon)[1, :])):
        dist = np.hstack((dist, [la.norm((x - x_moon)[:, i])]))
    a = min(dist)
    ai = np.where(dist == a)

    x_c_m_2, y_c_m_2 = circle(R_moon, (x_moon[0, ai[0]], x_moon[1, ai[0]]))
    axs.plot(x_c_m_2, y_c_m_2, color = "purple", linewidth = 1.5)

    axs.plot(x[0, ai], x[1, ai], color = "yellow", marker = "x", markersize = 5)

    axs.set_xlim(left = -5 * 10 ** 8, right = 5 * 10 ** 8)
    axs.set_ylim(bottom = -5 * 10 ** 8, top = 5 * 10 ** 8)
    axs.set_xlabel("")
    axs.set_ylabel("")
    fig.show()
    fig.savefig("Trajectory_simulation_NM")


if __name__ == "__main__":
    angle = 21.89719394321709
    magnitude = 3928.6754761701295
    norm_dv = np.array([[np.cos(np.deg2rad(angle))],
                        [np.sin(np.deg2rad(angle))],
                        [0]])
    dv_man = norm_dv * magnitude
    x, x_moon = Simulate(SimplePropagator, 0, 120, 400000, dv_man, dt = 100)
    dist = np.array([])
    for i in range(len((x)[1, :])):
        dist = np.hstack((dist, [la.norm((x)[:, i])]))
    a = min(dist)
    display(x, x_moon)
