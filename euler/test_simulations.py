import pdb

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sbpy.euler.animation import (animate_pressure, animate_solution,
                                  animate_velocity)
from sbpy.grid2d import MultiblockGrid, MultiblockSBP, get_center
from sbpy.utils import (create_convergence_table, export_to_tecplot,
                        get_bump_grid, get_channel_grid,
                        get_circle_sector_grid, get_gauss_initial_data)
from scipy.stats import multivariate_normal

# import mms_koaszany as mms
import mms_abl as mms
from euler import (euler_operator, force_operator, inflow_operator,
                   outflow_operator, pressure_operator, solve,
                   solve_steady_state, solve_steady_state_newton_krylov,
                   stabilized_natural_operator, wall_operator)

#import cProfile
# import mms


def solution_to_file(grid, U, V, P, name_base):

    for i in range(len(U)):
        filename = name_base+str(i)+'.tec'
        export_to_tecplot(grid,U[i],V[i],P[i],filename)

def bump_const_inflow_pressure_outflow(
        N = 30,
        num_timesteps = 10,
        dt = 0.1, 
        e = 0):

    X,Y = get_bump_grid(N)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)
    initu = np.array([np.ones(X.shape)])
    initv = np.array([np.zeros(X.shape)])
    initp = np.array([np.ones(X.shape)])
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()

    def spatial_op(state):
        S,J,ux,uy,vx,vy = euler_operator(sbp, state, e)

        Sbd,Jbd   = inflow_operator(sbp, state,  0, 'w', -1, 0, e) + \
                    pressure_operator(sbp, state, 0, 'e', ux, uy, vx, vy, e) + \
                    wall_operator(sbp, state, 0, 's', e) + \
                    wall_operator(sbp, state, 0, 'n', e)

        return S+Sbd, J+Jbd

    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt


def bump_const_inflow_and_outflow(
        N = 30,
        num_timesteps = 10,
        dt = 0.1,
        e = 0):

    X,Y = get_bump_grid(N)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)
    initu = np.array([np.ones(X.shape)])
    initv = np.array([np.zeros(X.shape)])
    initp = np.array([np.ones(X.shape)])
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()

    def spatial_op(state):
        S,J,ux,uy,vx,vy = euler_operator(sbp, state, e)

        Sbd,Jbd   = inflow_operator(sbp, state,  0, 'w', -1, 0, e) + \
                    stabilized_natural_operator(sbp,state, 0, 'e', 
                                                ux, uy, vx, vy, e) + \
                    wall_operator(sbp, state, 0, 's', e) + \
                    wall_operator(sbp, state, 0, 'n', e)

        return S+Sbd, J+Jbd

    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt


def bump_walls_and_outflow(
        N = 30,
        num_timesteps = 10,
        dt = 0.1, 
        e = 0):

    X,Y = get_bump_grid(N)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)
    initu, initv, initp = get_gauss_initial_data(X,Y,-0.5,0.4)
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()

    def spatial_op(state):
        S,J,ux,uy,vx,vy = euler_operator(sbp, state, e)

        Sbd,Jbd   = wall_operator(sbp, state,  0, 'w') + \
                    outflow_operator(sbp, state, 0, 'e', ux, uy, vx, vy, e) + \
                    wall_operator(sbp, state, 0, 's') + \
                    wall_operator(sbp, state, 0, 'n')

        return S+Sbd, J+Jbd

    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt


def square_walls_and_outflow(
        N = 30,
        num_timesteps = 10,
        dt = 0.1, 
        e = 0):

    (X,Y) = np.meshgrid(np.linspace(0,1,N), np.linspace(0,1,N))
    X = np.transpose(X)
    Y = np.transpose(Y)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)
    initu, initv, initp = get_gauss_initial_data(X,Y,0.5,0.5)
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()

    def spatial_op(state):
        S,J,ux,uy,vx,vy = euler_operator(sbp, state, e)

        Sbd,Jbd   = wall_operator(sbp, state,  0, 'w') + \
                    outflow_operator(sbp, state, 0, 'e', ux, uy, vx, vy, e) + \
                    wall_operator(sbp, state, 0, 's') + \
                    wall_operator(sbp, state, 0, 'n')

        return S+Sbd, J+Jbd

    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt


def square_outflow_everywhere(
        N = 30,
        num_timesteps = 10,
        dt = 0.1, e = 0):

    (X,Y) = np.meshgrid(np.linspace(0,1,N), np.linspace(0,1,N))
    X = np.transpose(X)
    Y = np.transpose(Y)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)
    initu, initv, initp = get_gauss_initial_data(X,Y,0.5,0.5)
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()

    def spatial_op(state):
        S,J,ux,uy,vx,vy = euler_operator(sbp, state, e)

        Sbd,Jbd   = outflow_operator(sbp, state,  0, 'w', ux, uy, vx, vy, e) + \
                    outflow_operator(sbp, state, 0, 'e', ux, uy, vx, vy, e) + \
                    outflow_operator(sbp, state, 0, 's', ux, uy, vx, vy, e) + \
                    outflow_operator(sbp, state, 0, 'n', ux, uy, vx, vy, e)

        return S+Sbd, J+Jbd

    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt


def circle_sector_outflow_everywhere(
        N = 30,
        num_timesteps = 10,
        dt = 0.1,
        angle = 0.5*np.pi,
        e = 0):

    (X,Y) = get_circle_sector_grid(N, 0.0, angle, 0.2, 1.0)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)
    initu, initv, initp = get_gauss_initial_data(X,Y,0.4,0.4)
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()

    def spatial_op(state):
        S,J,ux,uy,vx,vy = euler_operator(sbp, state, e)

        Sbd,Jbd   = outflow_operator(sbp, state,  0, 'w', ux, uy, vx, vy, e) + \
                    outflow_operator(sbp, state, 0, 'e', ux, uy, vx, vy, e) + \
                    outflow_operator(sbp, state, 0, 's', ux, uy, vx, vy, e) + \
                    outflow_operator(sbp, state, 0, 'n', ux, uy, vx, vy, e)

        return S+Sbd, J+Jbd

    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt


def square_cavity_flow(
        N = 30,
        num_timesteps = 10,
        dt = 0.1,
        e = 0):

    (X,Y) = np.meshgrid(np.linspace(0,1,N), np.linspace(0,1,N))
    X = np.transpose(X)
    Y = np.transpose(Y)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)

    initu, initv, initp = get_gauss_initial_data(X,Y,0.5,0.5)
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()

    def spatial_op(state):
        S,J = euler_operator(sbp, state, e)

        Sbd,Jbd   = wall_operator(sbp, state, 0, 'w', e) + \
                    wall_operator(sbp, state, 0, 'e', e) + \
                    wall_operator(sbp, state, 0, 's', e) + \
                    wall_operator(sbp, state, 0, 'n', e)

        return S+Sbd, J+Jbd


    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt


def circle_sector_cavity_flow(
        N = 30,
        num_timesteps = 10,
        dt = 0.1,
        angle = 0.5*np.pi,
        e = 0):


    (X,Y) = get_circle_sector_grid(N, 0.0, angle, 0.2, 1.0)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)

    initu, initv, initp = get_gauss_initial_data(X,Y,0.4,0.4)
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()


    def spatial_op(state):
        S,J,ux,uy,vx,vy = euler_operator(sbp, state, e)

        Sbd,Jbd   = wall_operator(sbp, state, 0, 'w', e) + \
                    wall_operator(sbp, state, 0, 'e', e) + \
                    wall_operator(sbp, state, 0, 's', e) + \
                    wall_operator(sbp, state, 0, 'n', e)

        return S+Sbd, J+Jbd

    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt

def bumpy_channel_flow(
        N = 30,
        num_timesteps = 10,
        dt = 0.1,
        e = 0):

    X,Y = get_channel_grid(N, -15, 10)
    #X,Y = get_bump_grid(N)

    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=4)

    initu = np.array([np.ones(X.shape)])
    initv = np.array([np.zeros(X.shape)])
    initp = np.array([np.zeros(X.shape)])

    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()

    turbulence = True

    def spatial_op(state):
        S,J,ux,uy,vx,vy = euler_operator(sbp, state, e, turbulence)

        Sbd,Jbd   = inflow_operator(sbp, state,  0, 'w',-1,0, e, turbulence) + \
                    pressure_operator(sbp, state, 0, 'e', 
                                     ux,uy,vx,vy, e, 0, 0, turbulence) + \
                    wall_operator(sbp, state, 0, 's', e, turbulence) + \
                    wall_operator(sbp, state, 0, 'n', e, turbulence)

        return S+Sbd, J+Jbd


    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P

def circle_sector_walls_and_outflow(
        N = 30,
        num_timesteps = 10,
        dt = 0.1,
        angle = 0.5*np.pi,
        e = 0):

    (X,Y) = get_circle_sector_grid(N, 0.0, angle, 0.2, 1.0)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=2)
    initu, initv, initp = get_gauss_initial_data(X,Y,0.4,0.4)
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()
    

    def spatial_op(state):
        S,J,ux,uy,vx,vy = euler_operator(sbp, state, e)

        Sbd,Jbd   = wall_operator(sbp, state,  0, 'w', e) + \
                    wall_operator(sbp, state, 0, 'e', e) + \
                    wall_operator(sbp, state, 0, 's', e) + \
                    outflow_operator(sbp, state, 0, 'n', ux, uy, vx, vy, e)

        return S+Sbd, J+Jbd

    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt


#SELECT ONE OF THE FUNCTIONS BELOW AND RUN SCRIPT
#-----------------------------------
#bump_const_inflow_pressure_outflow()
#bump_const_inflow_and_outflow()
#bump_walls_and_outflow()
#square_walls_and_outflow()
#square_outflow_everywhere()
#circle_sector_outflow_everywhere()
#square_cavity_flow()
#circle_sector_cavity_flow()

if __name__ == '__main__':

    # for mms
    grid,U,V,P = bumpy_channel_flow( N = 60, dt = 0.05, e = 0.001, num_timesteps = 400)
    solution_to_file(grid,U,V,P,'plots2/movie')
