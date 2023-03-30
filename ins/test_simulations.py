import pdb
#import cProfile

from scipy.stats import multivariate_normal
import scipy
import numpy as np
import matplotlib.pyplot as plt

from sbpy.utils import get_circle_sector_grid, get_bump_grid, get_channel_grid, create_convergence_table
from sbpy.grid2d import MultiblockGrid, MultiblockSBP, get_center
from sbpy.ins.animation import animate_pressure, animate_velocity, animate_solution
from ins import ins_operator, wall_operator, outflow_operator, pressure_operator, inflow_operator, implicit_integration, force_operator

def solution_to_file(grid,U,V,P,name_base):

    for i in range(len(U)):
        filename = name_base+str(i)+'.tec'
        export_to_tecplot(grid,U[i],V[i],P[i],filename)


def get_gauss_initial_data(X, Y, cx, cy):
    rv1 = multivariate_normal([cx,cy], 0.01*np.eye(2))
    gauss_bell = rv1.pdf(np.dstack((X,Y)))
    normalize = np.max(gauss_bell.flatten())
    initu = 2*np.array([gauss_bell])/normalize
    initv = np.array([np.zeros(X.shape)])
    initp = np.array([np.ones(X.shape)])

    return initu, initv, initp


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
        S,J,ux,uy,vx,vy = ins_operator(sbp, state, e)

        Sbd,Jbd   = inflow_operator(sbp, state,  0, 'w', -1, 0, e) + \
                    pressure_operator(sbp, state, 0, 'e', ux, uy, vx, vy, e) + \
                    wall_operator(sbp, state, 0, 's', e) + \
                    wall_operator(sbp, state, 0, 'n', e)

        return S+Sbd, J+Jbd

    U,V,P = implicit_integration(grid, spatial_op, initu,
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
        S,J,ux,uy,vx,vy = ins_operator(sbp, state, e)

        Sbd,Jbd   = inflow_operator(sbp, state,  0, 'w', -1, 0, e) + \
                    outflow_operator(sbp,state, 0, 'e', ux, uy, vx, vy, e) + \
                    wall_operator(sbp, state, 0, 's', e) + \
                    wall_operator(sbp, state, 0, 'n', e)

        return S+Sbd, J+Jbd

    U,V,P = implicit_integration(grid, spatial_op, initu,
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
        S,J,ux,uy,vx,vy = ins_operator(sbp, state, e)

        Sbd,Jbd   = wall_operator(sbp, state,  0, 'w') + \
                    outflow_operator(sbp, state, 0, 'e', ux, uy, vx, vy, e) + \
                    wall_operator(sbp, state, 0, 's') + \
                    wall_operator(sbp, state, 0, 'n')

        return S+Sbd, J+Jbd

    U,V,P = implicit_integration(grid, spatial_op, initu,
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
        S,J,ux,uy,vx,vy = ins_operator(sbp, state, e)

        Sbd,Jbd   = wall_operator(sbp, state, 0, 'w', e) + \
                    wall_operator(sbp, state, 0, 'e', e) + \
                    wall_operator(sbp, state, 0, 's', e) + \
                    wall_operator(sbp, state, 0, 'n', e)

        return S+Sbd, J+Jbd


    U,V,P = implicit_integration(grid, spatial_op, initu,
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

    #for time simulation
    grid,U,V,P,dt = square_cavity_flow(N = 40, num_timesteps = 20, dt = 1e-1, e = 0)

