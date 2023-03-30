import pdb
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sbpy.grid2d import MultiblockGrid, MultiblockSBP
from sbpy.utils import (create_convergence_table, export_to_tecplot, surf_plot,
                       get_gauss_initial_data, nice_plot)

import sbpy.operators # for post_process_fine_solution
from sbpy.artificial_dissipation import generate_artificial_dissipation_matrix_2D
import matplotlib.pyplot as plt
from matplotlib import rc,cm
import scipy
from random import random
from euler import (euler_operator, force_operator, inflow_operator,
                   outflow_operator, pressure_operator,
                   solve, solve_steady_state, wall_operator,
                   solve_steady_state_newton_krylov, interior_penalty_dirichlet_operator, 
                   interior_penalty_neumann_operator, interior_penalty_robin_operator)

from sbpy.abl_utils import (evaluate_1d_in_y_function_on_grid, bd_func_damping, 
                            bd_func_no_damping, read_ins_data, get_plus_variables, 
                            compute_u_tau, write_ins_data, get_solution_slice,
                            compute_low_error_slice, compute_uy_lw_error_slice,
                            post_process_fine_solution, slice_to_full_grid_function, 
                            compute_low_error_robin)

def steady_state_couette_flow(N, e = 1e-4, acc_y = 4):
    middle = 1
    Nx    = 4
    y = np.linspace(0,2,N)
    (X,Y) = np.meshgrid(np.linspace(0,1,Nx), y)
    X     = np.transpose(X)
    Y     = np.transpose(Y)
    X     = X[:-1]
    Y     = Y[:-1]
 
    grid = MultiblockGrid([(X,Y)])
    sbp  = MultiblockSBP(grid, accuracy_x = 2, accuracy_y = acc_y, periodic = True)

    speed = 2
    #initu = [Y + 0.1*np.random.rand(X.shape[0],X.shape[1])]
    initu = [Y]#np.array([np.zeros(X.shape)])
    initv = np.array([np.zeros(X.shape)])
    initp = np.array([np.zeros(X.shape)])

    def spatial_op(state):
        turbulence = True
        u_tau = 0.044546384775524965
        u_tau = 0.044539130430358315
        bd_func_1d = lambda y: bd_func_damping(y, e, u_tau, 1)
        lm_bd_func = lambda y: evaluate_1d_in_y_function_on_grid(y, bd_func_1d)
        S,J,ux,uy,vx,vy = euler_operator(sbp, state, e, turb_model = turbulence, 
                                         lm_func = lm_bd_func)

        Sbd,Jbd = inflow_operator(sbp,state,0,'n', 0, -speed, e, turb_model = turbulence, 
                                   lm_func = lm_bd_func) +\
                  wall_operator(sbp,state,0,'s',e)
                  #inflow_operator(sbp,state,0,'s', 0, 0, e, turb_model = turbulence, 
                  #                 lm_func = lm_bd_func)
        return S+Sbd, J+Jbd
    dt = 1e-3
    num_steps = 100
    #U,V,P = solve(grid, spatial_op, initu, initv, initp, dt, num_steps, sbp)
    U,V,P = solve_steady_state(grid, spatial_op, initu, initv, initp)
    #U,V,P = solve_steady_state_newton_krylov(grid, spatial_op, initu, initv, initp)
    #grid.plot_grid_function(U[-1])
    return sbp,U


def run_fine_simulation(N = 1000, e = 1/10000):
    #sbp,U = steady_state_couette_flow(N, e, acc_y = 4)
    #blocks = sbp.grid.get_blocks()
    #Y = blocks[0][1]
    #u = U[-1][2]
    #y = Y[1]
    file_name = "data_files/fine_solution_wall_dirichlet.dat"
    y, u = read_ins_data(file_name)

    u_tau = 0.044546384775524965
    u_tau = 0.044539130430358315
    kappa = 0.33
    C = 4.1
    D = u_tau*(1/kappa*np.log(u_tau/e) + C)
    plt.semilogx(y, u_tau*(1/kappa*np.log(y)) + D)
    plt.plot(y,u)
    plt.show()
    print(compute_u_tau(y,u,e))
    return y, u

if __name__ == '__main__':
    e = 1/10000
    y, u = run_fine_simulation()
    #plt.semilogx(y,u)
    #plt.show()
    #file_name = "data_files/fine_solution_wall_dirichlet.dat"
    #y_fine, u_fine = read_ins_data(file_name)
    #post_process_fine_solution(file_name, e)
    #pdb.set_trace()
