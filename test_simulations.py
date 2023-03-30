import pdb
#import cProfile

from scipy.stats import multivariate_normal
import scipy
import numpy as np
import matplotlib.pyplot as plt

from sbpy.utils import get_circle_sector_grid, get_bump_grid, get_channel_grid, create_convergence_table, export_to_tecplot
from sbpy.grid2d import MultiblockGrid, MultiblockSBP, get_center
from sbpy.euler.animation import animate_pressure, animate_velocity, animate_solution
from euler import euler_operator, wall_operator, outflow_operator, pressure_operator, inflow_operator, solve, solve_steady_state, stabilized_natural_operator, solve_steady_state_newton_krylov, force_operator

#import mms_koaszany as mms
import mms_abl as mms
#import mms

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def solution_to_file(grid,U,V,P,name_base):

    for i in range(len(U)):
        filename = name_base+str(i)+'.tec'
        export_to_tecplot(grid,U[i],V[i],P[i],filename)


def get_gauss_initial_data(X, Y, cx, cy):
    rv1 = multivariate_normal([cx,cy], 0.01*np.eye(2))
    gauss_bell = rv1.pdf(np.dstack((X,Y)))
    normalize = np.max(gauss_bell.flatten())
    initu = 2*np.array([gauss_bell])/normalize
    #initu = np.array([np.sin(2*np.pi*X)*np.sin(2*np.pi*Y)])
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
                    outflow_operator(sbp,state, 0, 'e', ux, uy, vx, vy, e) + \
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
    initu = initp
    initv = initp
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()

    def spatial_op(state):
        S,J,ux,uy,vx,vy = euler_operator(sbp, state, e)

        Sbd,Jbd   = wall_operator(sbp, state, 0, 'w', e) + \
                    wall_operator(sbp, state, 0, 'e', e) + \
                    wall_operator(sbp, state, 0, 's', e) + \
                    inflow_operator(sbp, state, 0, 'n', 0,-1,e)

        return S+Sbd, J+Jbd


    U,V,P = solve_steady_state(grid, spatial_op, initu,
                  initv, initp)

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

    X,Y = get_bump_grid(N)

    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy=4)

    initu = np.array([np.zeros(X.shape)])
    initv = np.array([np.zeros(X.shape)])
    initp = np.array([np.zeros(X.shape)])

    grid.plot_grid()
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()

    y = Y[0]
    wn_data = 1 - np.tanh(50*y) + np.tanh(50*(y-0.8))

    def spatial_op(state):
        S,J,ux,uy,vx,vy = euler_operator(sbp, state, e)

        Sbd,Jbd   = inflow_operator(sbp, state,  0, 'w',wn_data,0, e) + \
                    outflow_operator(sbp, state, 0, 'e', ux,uy,vx,vy, e) + \
                    wall_operator(sbp, state, 0, 's', e) + \
                    wall_operator(sbp, state, 0, 'n', e)

        return S+Sbd, J+Jbd


    U,V,P = solve(grid, spatial_op, initu,
                  initv, initp, dt, num_timesteps)

    return grid,U,V,P,dt

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


def steady_state_mms(acc = 2):

    # need import mms_file as mms on top  to run
    # remember: epsilon in mms-file and here need to agree!

    n_vec = np.array([25, 35])
    e     = mms.e

    err_vec = []

    for N in n_vec:
        print(N)
    
        #X,Y = get_channel_grid(N,x0 = -0.5, x1 = 1)
        (X,Y) = np.meshgrid(np.linspace(0,1,N), np.linspace(0,2,N))
        X = np.transpose(X)
        Y = np.transpose(Y)

        grid = MultiblockGrid([(X,Y)])
        sbp = MultiblockSBP(grid, accuracy = acc)

        initu = np.array([4*np.ones(X.shape)])
        initv = np.array([np.ones(X.shape)])
        initp = np.array([np.ones(X.shape)])

        def spatial_op(state):
            t = 0
            S,J,ux,uy,vx,vy = euler_operator(sbp, state, e)

            
            Sbd,Jbd = inflow_operator(sbp,state,0,'w',\
                                mms.wn_data(sbp,0,'w', t), \
                                mms.wt_data(sbp,0,'w',t),e) + \
                      pressure_operator(sbp, state, 0, 'e', ux, uy, vx, vy,\
                                mms.normal_outflow_data(sbp,0,'e',t), \
                                mms.tangential_outflow_data(sbp,0,'e',t), e) + \
                      wall_operator(sbp, state, 0,'s', e) + \
                      pressure_operator(sbp, state, 0, 'n', ux, uy, vx, vy,\
                                mms.normal_outflow_data(sbp,0,'n',t), \
                                mms.tangential_outflow_data(sbp,0,'n',t), e) 
            S -= force_operator(sbp,mms.force1,mms.force2,mms.force3,t)[0]

            return S+Sbd, J+Jbd
        U,V,P = solve_steady_state(grid, spatial_op, initu, initv,initp)

        err = np.array([U[-1]-mms.u(0,X,Y), V[-1]-mms.v(0,X,Y), \
                       P[-1] - mms.p(0,X,Y)])
        err = err.flatten()
        block_idx = 0
        P_big = scipy.sparse.kron(scipy.sparse.eye(3),sbp.get_full_P(block_idx))
        err = np.sqrt(np.transpose(err)@P_big@err)
        err_vec.append(err)

    err_vec = np.array(err_vec)
    create_convergence_table(n_vec,err_vec, 1/(n_vec-1))

    return grid,U,V,P


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
    grid,U,V,P,dt = bumpy_channel_flow(N = 80, num_timesteps = 500, dt = 0.5e-2, 
                                       e = 1e-3)
    #grid,U,V,P,dt = square_cavity_flow(N = 80, num_timesteps = 100, dt = 1e-2,e=1e-3)

    # for mms
    #grid,U,V,P = steady_state_mms(acc = 4)
    #solution_to_file(grid,U,V,P,'plots/movie')

    grid.plot_grid_function(U[-1])
