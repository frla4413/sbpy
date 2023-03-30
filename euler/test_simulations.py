import unittest

from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt

from sbpy.utils import get_circle_sector_grid, get_bump_grid
from sbpy.grid2d import MultiblockGrid, MultiblockSBP, get_center
from sbpy.euler.animation import animate_pressure, animate_velocity, animate_solution
from euler import euler_operator, wall_operator, outflow_operator, pressure_operator, inflow_operator, outflow_operator, solve, solve_steady_state


def get_gauss_initial_data(X, Y, cx, cy):
    rv1 = multivariate_normal([cx,cy], 0.01*np.eye(2))
    gauss_bell = rv1.pdf(np.dstack((X,Y)))
    normalize = np.max(gauss_bell.flatten())
    initu = 2*np.array([gauss_bell])/normalize
    initv = np.array([np.zeros(X.shape)])
    initp = np.array([np.ones(X.shape)])

    return initu, initv, initp


def get_gresho_initial_data(X, Y):

    size = X.shape
    X = X.flatten()
    Y = Y.flatten()
    r = np.sqrt(X**2 + Y**2)

    initu = []
    initv = []
    initp = []

    C2 = (-12.5)*0.4**2 + 20*0.4**2 - 4*np.log(0.4)
    C1 = C2 - 20*0.2 + 4*np.log(0.2)
    for k in range(size[0]*size[1]):
        rk = r[k]
        xk = X[k]
        yk = Y[k]

        if rk <= 0.2:
            initu.append(-5*yk)
            initv.append(5*xk)
            initp.append(12.5*rk**2 + C1)
        elif rk > 0.4:
            initu.append(0)
            initv.append(0)
            initp.append(0)
        else:
            initu.append(-2*yk/rk + 5*yk)
            initv.append(2*xk/rk - 5*xk)
            initp.append(12.5*rk**2 - 20*rk + 4*np.log(rk) + C2)


    initu = np.array([np.reshape(np.array(initu),size)])
    initv = np.array([np.reshape(np.array(initv),size)])
    initp = np.array([np.reshape(np.array(initp),size)])
    return initu, initv, initp


def bump_const_inflow_pressure_outflow(
        N = 30,
        num_timesteps = 10,
        dt = 0.1):

    X,Y = get_bump_grid(N)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)
    initu = np.array([np.ones(X.shape)])
    initv = np.array([np.zeros(X.shape)])
    initp = np.array([np.ones(X.shape)])
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()


    def spatial_op(state):
        S,J = euler_operator(sbp, state) + \
              inflow_operator(sbp, state, 0, 'w', -1, 0) + \
              pressure_operator(sbp, state, 0, 'e') + \
              wall_operator(sbp, state, 0, 's') + \
              wall_operator(sbp, state, 0, 'n')

        return S, J


    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt


def bump_const_inflow_pressure_speed_outflow(
        N = 30,
        num_timesteps = 10,
        dt = 0.1):

    X,Y = get_bump_grid(N)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)
    initu = np.array([np.ones(X.shape)])
    initv = np.array([np.zeros(X.shape)])
    initp = np.array([np.ones(X.shape)])
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()


    def spatial_op(state):
        S,J = euler_operator(sbp, state) + \
              inflow_operator(sbp, state, 0, 'w', -1, 0) + \
              outflow_operator(sbp, state, 0, 'e') + \
              wall_operator(sbp, state, 0, 's') + \
              wall_operator(sbp, state, 0, 'n')

        return S, J


    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt


def bump_walls_and_pressure_speed_outflow(
        N = 30,
        num_timesteps = 10,
        dt = 0.1):

    X,Y = get_bump_grid(N)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)
    initu, initv, initp = get_gauss_initial_data(X,Y,-0.5,0.4)
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()


    def spatial_op(state):
        S,J = euler_operator(sbp, state) + \
              wall_operator(sbp, state, 0, 'w') + \
              outflow_operator(sbp, state, 0, 'e') + \
              wall_operator(sbp, state, 0, 's') + \
              wall_operator(sbp, state, 0, 'n')

        return S, J


    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt


def square_walls_and_pressure_speed_outflow(
        N = 30,
        num_timesteps = 10,
        dt = 0.1):

    (X,Y) = np.meshgrid(np.linspace(0,1,N), np.linspace(0,1,N))
    X = np.transpose(X)
    Y = np.transpose(Y)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)
    initu, initv, initp = get_gauss_initial_data(X,Y,0.5,0.5)
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()


    def spatial_op(state):
        S,J = euler_operator(sbp, state) + \
              wall_operator(sbp, state, 0, 'w') + \
              outflow_operator(sbp, state, 0, 'e') + \
              wall_operator(sbp, state, 0, 's') + \
              wall_operator(sbp, state, 0, 'n')

        return S, J


    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt


def square_pressure_speed_outflow_everywhere(
        N = 30,
        num_timesteps = 10,
        dt = 0.1):

    (X,Y) = np.meshgrid(np.linspace(0,1,N), np.linspace(0,1,N))
    X = np.transpose(X)
    Y = np.transpose(Y)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)
    initu, initv, initp = get_gauss_initial_data(X,Y,0.5,0.5)
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()


    def spatial_op(state):
        S,J = euler_operator(sbp, state) + \
              outflow_operator(sbp, state, 0, 'w') + \
              outflow_operator(sbp, state, 0, 'e') + \
              outflow_operator(sbp, state, 0, 's') + \
              outflow_operator(sbp, state, 0, 'n')

        return S, J


    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt


def circle_sector_pressure_speed_outflow_everywhere(
        N = 30,
        num_timesteps = 10,
        dt = 0.1,
        angle = 0.5*np.pi):

    (X,Y) = get_circle_sector_grid(N, 0.0, angle, 0.2, 1.0)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)
    initu, initv, initp = get_gauss_initial_data(X,Y,0.4,0.4)
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()


    def spatial_op(state):
        S,J = euler_operator(sbp, state) + \
              outflow_operator(sbp, state, 0, 'w') + \
              outflow_operator(sbp, state, 0, 'e') + \
              outflow_operator(sbp, state, 0, 's') + \
              outflow_operator(sbp, state, 0, 'n')

        return S, J


    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt


def square_cavity_flow(
        N = 30,
        num_timesteps = 10,
        dt = 0.1):

    (X,Y) = np.meshgrid(np.linspace(0,1,N), np.linspace(0,1,N))
    X = np.transpose(X)
    Y = np.transpose(Y)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)

    initu, initv, initp = get_gauss_initial_data(X,Y,0.5,0.5)
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()


    def spatial_op(state):
        S,J = euler_operator(sbp, state) + \
              wall_operator(sbp, state, 0, 'w') + \
              wall_operator(sbp, state, 0, 'e') + \
              wall_operator(sbp, state, 0, 's') + \
              wall_operator(sbp, state, 0, 'n')

        return S, J


    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt

def gresho_test(
        N = 30,
        num_timesteps = 10,
        dt = 0.1):

    (X,Y) = np.meshgrid(np.linspace(-0.5,0.5,N),
                        np.linspace(-0.5,0.5,N))
    X = np.transpose(X)
    Y = np.transpose(Y)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)

    initu, initv, initp = get_gresho_initial_data(X, Y)
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()

    def spatial_op(state):

         S,J = euler_operator(sbp, state) + \
              wall_operator(sbp, state, 0, 'w') + \
              wall_operator(sbp, state, 0, 'e') + \
              wall_operator(sbp, state, 0, 's') + \
              wall_operator(sbp, state, 0, 'n')
         return S,J

    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    e = []
    for u,v in zip(U,V):
        w = u**2 + v**2
        e.append(sbp.integrate([w])/2)

    t = []
    for k in range(num_timesteps):
        t.append(k*dt)

    plt.plot(t,e)
    plt.show()

    return grid,U,V,P,dt



def circle_sector_cavity_flow(
        N = 30,
        num_timesteps = 10,
        dt = 0.1,
        angle = 0.5*np.pi):

    (X,Y) = get_circle_sector_grid(N, 0.0, angle, 0.2, 1.0)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)

    initu, initv, initp = get_gauss_initial_data(X,Y,0.4,0.4)
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()


    def spatial_op(state):
        S,J = euler_operator(sbp, state) + \
              wall_operator(sbp, state, 0, 'w') + \
              wall_operator(sbp, state, 0, 'e') + \
              wall_operator(sbp, state, 0, 's') + \
              wall_operator(sbp, state, 0, 'n')

        return S, J


    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt

def bump_const_inflow_pressure_steady_state( N = 30):

    X,Y = get_bump_grid(N)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)
    initu = np.array([np.ones(X.shape)])
    initv = np.array([np.zeros(X.shape)])
    initp = np.array([np.ones(X.shape)])
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()


    def spatial_op(state):
        S,J = euler_operator(sbp, state) + \
              inflow_operator(sbp, state, 0, 'w', -1, 0) + \
              pressure_operator(sbp, state, 0, 'e') + \
              wall_operator(sbp, state, 0, 's') + \
              wall_operator(sbp, state, 0, 'n')

        return S, J


    U,V,P = solve_steady_state(grid, spatial_op, initu,
                  initv, initp)

    dt = 0.1

    return grid,U,V,P,dt


#SELECT ONE OF THE FUNCTIONS BELOW AND RUN SCRIPT
#-----------------------------------
#bump_const_inflow_pressure_outflow()
#bump_const_inflow_pressure_speed_outflow()
#bump_walls_and_pressure_speed_outflow()
#square_walls_and_pressure_speed_outflow()
#square_pressure_speed_outflow_everywhere()
#circle_sector_pressure_speed_outflow_everywhere()
#square_cavity_flow()
#circle_sector_cavity_flow()

if __name__ == '__main__':
#    grid,U,V,P,dt = circle_sector_pressure_speed_outflow_everywhere(num_timesteps=15)

    grid,U,V,P,dt = gresho_test(N=48,num_timesteps=2000, dt=0.005)
    animate_velocity(grid,U,V,dt)

    # for steady state simulation
    #grid,U,V,P,dt = bump_const_inflow_pressure_steady_state(30)
    #animate_velocity(grid,U,V,dt)
    #np.savetxt('bump_results/bump_u.txt',U[-1])
    #np.savetxt('bump_results/bump_v.txt',V[-1])
    #np.savetxt('bump_results/bump_p.txt',P[-1])
    #X,Y = grid.get_block(0)
    #np.savetxt('bump_results/X.txt',X)
    #np.savetxt('bump_results/Y.txt',Y)
