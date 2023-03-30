import pdb

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.cm import ScalarMappable

def animate_pressure(grid, P, dt):
    nt = len(P)
    fig, ax = plt.subplots(1,1)
    X,Y = grid.get_block(0)
    plot = ax.pcolormesh(X,Y,P[0])

    def update(num, pressure_plot):
        p = P[num%nt]
        pressure_plot = ax.pcolormesh(X,Y,p)
        print("t = {:.2f}".format((num%nt)*dt), end='\r')

        return pressure_plot,

    anim = animation.FuncAnimation(fig, update, fargs=(plot,), interval=1000*dt, blit = True)
    plt.show()


def animate_velocity(grid, U, V, dt):
    nt = len(U)
    fig, ax = plt.subplots(1,1)
    X,Y = grid.get_block(0)
    plot = ax.quiver(X,Y,U[0],V[0])

    def update(num, plot):
        u = U[num%nt]
        v = V[num%nt]
        plot.set_UVC(u,v)
        print("t = {:.2f}".format((num%nt)*dt), end='\r')

        return plot,

    anim = animation.FuncAnimation(fig, update, fargs=(plot,), interval=1000*dt, blit = True)
    plt.show()


def animate_solution(grid, U, V, P, dt):
    nt = len(U)
    fig, ax = plt.subplots(1,1)
    X,Y = grid.get_block(0)
    p_plot = ax.pcolormesh(X,Y,P[0])
    w_plot = ax.quiver(X,Y,U[0],V[0])
    p_min = np.min(np.array(P).flatten())
    p_max = np.max(np.array(P).flatten())
    p_plot.set_clim([p_min, p_max])
    fig.colorbar(p_plot, ax=ax)

    def update(num, p_plot, w_plot):
        u = U[num%nt]
        v = V[num%nt]
        p = P[num%nt]

        p_min = np.min(np.array(p).flatten())
        p_max = np.max(np.array(p).flatten())
        p_plot.set_clim([p_min, p_max])
        p_plot.set_array(p[:-1,:-1].ravel())
        w_plot.set_UVC(u,v)
        print("t = {:.2f}".format((num%nt)*dt), end='\r')

        return p_plot, w_plot

    anim = animation.FuncAnimation(fig, update, fargs=(p_plot, w_plot), interval=1000*dt, blit = True)
    plt.show()

