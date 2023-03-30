import pdb

import matplotlib.pyplot as plt
import numpy as np
import scipy
from sbpy.grid2d import MultiblockGrid, MultiblockSBP
from sbpy.utils import (create_convergence_table, export_to_tecplot, surf_plot,
                       get_gauss_initial_data, nice_plot)

import matplotlib.pyplot as plt
from matplotlib import rc,cm
import scipy
from euler import (euler_operator, force_operator, inflow_operator,
                   interior_low_operator, outflow_operator, pressure_operator,
                   solve, solve_steady_state, wall_operator,
                   solve_steady_state_newton_krylov,
                   interior_penalty_operator)

from sbpy.abl_utils import evaluate_1d_in_y_function_on_grid, bd_func_damping, bd_func_no_damping, read_ins_data, read_full_ins_data, get_plus_variables, compute_u_tau, write_ins_data, write_full_ins_data


def steady_state_couette_flow(N, e = 0.01, acc_y = 2, interior_penalty = False,
                              lw_slice = slice(None,None), data = 0, alpha = 1):
    middle = 1
    Nx    = 4
    (X,Y) = np.meshgrid(np.linspace(0,1,Nx), np.linspace(0,2*middle,N))
    X     = np.transpose(X)
    Y     = np.transpose(Y)
    X     = X[:-1]
    Y     = Y[:-1]
 
    grid = MultiblockGrid([(X,Y)])
    sbp  = MultiblockSBP(grid, accuracy_x = 2, accuracy_y = acc_y, periodic = True)

    speed = 1
    initu = -speed + speed*np.array([np.ones(X.shape)])
    initv = np.array([np.zeros(X.shape)])
    initp = np.array([np.zeros(X.shape)])

    if interior_penalty:
        lw_slicex = slice(None,None)
    def spatial_op(state):
        turbulence = True
        bd_func_1d = lambda y: bd_func_damping(y)
        lm_bd_func = lambda y: evaluate_1d_in_y_function_on_grid(y, bd_func_1d)
        S,J,ux,uy,vx,vy = euler_operator(sbp, state, e, turb_model = turbulence, 
                                         lm_func = lm_bd_func)

        Sbd,Jbd = inflow_operator(sbp,state,0,'s', 0, -speed, e, turb_model = turbulence, 
                                   lm_func = lm_bd_func) +\
                  inflow_operator(sbp,state,0,'n', 0, -speed, e, turb_model = turbulence, 
                                   lm_func = lm_bd_func)
        if interior_penalty:
            lw_slice1 = slice(None,None)
            lw_slice2 = lw_slice
            Si, Ji = interior_penalty_operator(sbp, state, 0, lw_slice1, 
                     lw_slice2, data)
            S += Si
            J += Ji
        return S+Sbd, J+Jbd

    U,V,P = solve_steady_state(grid, spatial_op, initu, initv, initp, alpha)
    return sbp,U

def run_fine_simulation(N = 1000, e = 1/10000):
    
    sbp,U = steady_state_couette_flow(N, e, acc_y = 2, alpha = 1)
    blocks = sbp.grid.get_blocks()
    Y = blocks[0][1]
    u = U[-1][2]
    y = Y[1]

    plt.semilogx(y,u)
    plt.show()
    write_full_ins_data("data_files/fine_full_solution.dat", Y, U[-1])
    

def couette_mpt(N = 100, e = 1/10000):

    jump = 20
    Y_fine, u_fine = read_full_ins_data("data_files/fine_full_solution.dat")
    u_fine_on_coarse_grid = u_fine[:,0::jump]
    print("Red fine data!")
    
    lw_slice = slice(10,20)
    sbp,U = steady_state_couette_flow(N, e, acc_y = 2, 
     lw_slice=lw_slice, interior_penalty=True,data = u_fine_on_coarse_grid, alpha = 0.1)
    blocks = sbp.grid.get_blocks()
    Y = blocks[0][1]
    u_coarse_mpt = U[-1]
    y = Y[1]

    ind = len(y)
    y = y[0:ind]
    
    plt.rcParams.update({'font.size': 35})
    fig = plt.figure(figsize=(11,9))
    #fig2 = plt.figure(figsize=(11,9))
    ax  = fig.gca()
    ax.semilogx(y, u_coarse_mpt[1,0:ind])
    ax.semilogx(y, u_fine_on_coarse_grid[1,0:ind],'--k')
    ax.semilogx(y[lw_slice], u_fine_on_coarse_grid[0,0]*np.ones(y[lw_slice].shape),'xk')

    sbp,U = steady_state_couette_flow(N, e, acc_y = 2, alpha = 0.2)
    u_coarse_no_mpt = U[-1]
    ax.semilogx(y, u_coarse_no_mpt[1,0:ind],':')

    err_mpt = np.linalg.norm(u_fine_on_coarse_grid - u_coarse_mpt)
    err_no_mpt = np.linalg.norm(u_fine_on_coarse_grid - u_coarse_no_mpt)
    print(err_mpt, err_no_mpt)
    err_mpt_slice = np.linalg.norm(u_fine_on_coarse_grid[1,lw_slice] - u_coarse_mpt[1,lw_slice])
    err_no_mpt_slice = np.linalg.norm(u_fine_on_coarse_grid[1,lw_slice] - u_coarse_no_mpt[1,lw_slice])
    print(err_mpt_slice, err_no_mpt_slice)
    plt.show()

if __name__ == '__main__':
    #run_fine_simulation(N = 2001, e = 1/10000)
    couette_mpt(N = 101, e = 1/10000)
