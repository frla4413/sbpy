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

def steady_state_couette_flow(N, e = 0.01, acc_y = 2, interior_penalty = False,
                              lw_slice = slice(None,None), beta = 0, alpha = 1,
                              a = 0):
    middle = 1
    Nx    = 4
    y = np.linspace(0,1,N)
    c = 3
    b1 = y**c
    b2 = (1 - y)**c
    y = b1/(b1 + b2)
    y = 2*y
    #y = np.linspace(0,2,N)
    (X,Y) = np.meshgrid(np.linspace(0,1,Nx), y)
    X     = np.transpose(X)
    Y     = np.transpose(Y)
    X     = X[:-1]
    Y     = Y[:-1]
 
    grid = MultiblockGrid([(X,Y)])
    sbp  = MultiblockSBP(grid, accuracy_x = 2, accuracy_y = acc_y, periodic = True)

    speed = 2
    initu = np.array([np.zeros(X.shape)])
    initv = np.array([np.zeros(X.shape)])
    initp = np.array([np.zeros(X.shape)])
    dissipation = True
    art_diss_matrix = generate_artificial_dissipation_matrix_2D(grid, alpha = 0.1)
    art_diss_matrix = scipy.sparse.kron(scipy.sparse.eye(3), art_diss_matrix)

    if interior_penalty:
        lw_slicex = slice(None,None)

    def spatial_op(state):
        turbulence = True
        u_tau = 0.044546384775524965
        bd_func_1d = lambda y: bd_func_damping(y, e, u_tau, 1)
        lm_bd_func = lambda y: evaluate_1d_in_y_function_on_grid(y, bd_func_1d)
        S,J,ux,uy,vx,vy = euler_operator(sbp, state, e, turb_model = turbulence, 
                                         lm_func = lm_bd_func)

        Sbd,Jbd = inflow_operator(sbp,state,0,'n', 0, -speed, e, turb_model = turbulence, 
                                   lm_func = lm_bd_func) +\
                  wall_operator(sbp,state,0,'s',e)
                  #inflow_operator(sbp,state,0,'s', 0, 0, e, turb_model = turbulence, 
                  #                 lm_func = lm_bd_func)
        if interior_penalty == 'robin':
            lw_slice1 = slice(None,None)
            lw_slice2 = lw_slice
            Si, Ji = interior_penalty_robin_operator(sbp, state, 0, lw_slice1, lw_slice2, uy, e, beta)
            S += Si
            J += Ji
        if interior_penalty == 'dirichlet':
            lw_slice1 = slice(None,None)
            lw_slice2 = lw_slice
            data = a*np.log(Y[:,1:]/beta)
            data = np.concatenate((np.zeros((Y.shape[0],1)), data), axis = 1)
            Si, Ji = interior_penalty_dirichlet_operator(sbp, state, 0, lw_slice1, 
                     lw_slice2, data)
            S += Si
            J += Ji

        if interior_penalty == 'neumann':
            lw_slice1 = slice(None,None)
            lw_slice2 = lw_slice
            data = np.zeros(Y.shape)
            data[:,1:] = a/Y[:,1:]
            Si, Ji = interior_penalty_neumann_operator(sbp, state, 0, lw_slice1, 
                     lw_slice2, data)
            S += Si
            J += Ji
        if dissipation: 
            S+= art_diss_matrix@state
            J+= art_diss_matrix
        return S+Sbd, J+Jbd

    #U,V,P = solve_steady_state(grid, spatial_op, initu, initv, initp, alpha)
    U,V,P = solve_steady_state_newton_krylov(grid, spatial_op, initu, initv, initp)
    return sbp,U


def run_fine_simulation(N = 1000, e = 1/10000, alpha = 1, file_name = "fine_solution.dat"):
    sbp,U = steady_state_couette_flow(N, e, acc_y = 4, alpha = alpha)
    blocks = sbp.grid.get_blocks()
    Y = blocks[0][1]
    u = U[-1][2]
    y = Y[1]
    write_ins_data(file_name, y, u)
    return y, u

def couette_no_mpt(N, e, alpha = 1, plot = False, file_name = None):
    sbp,U = steady_state_couette_flow(N, e, acc_y = 2, alpha = alpha)

    if plot:
        y, u, uy = get_solution_slice(sbp, U)
        y_fine, u_fine = read_ins_data(file_name)
        ind = int(len(u_fine))
        plt.rcParams.update({'font.size': 35})
        fig = plt.figure(figsize=(11,9))
        ax  = fig.gca()
        ax.semilogx(y_fine[:ind], u_fine[:ind], '--k')
        ind = int(len(u)) 
        ax.semilogx(y[:ind], u[:ind])
        plt.title("Fine and coarse solution")
        plt.show()
    return U, sbp


def couette_mpt(N, e, lw_slice, alpha = 1, plot = False, file_name = "fine_solution.dat"):
    beta, u_tau, a, b = post_process_fine_solution(file_name, e)
    sbp,U = steady_state_couette_flow(N, e, acc_y = 2, 
            lw_slice=lw_slice, interior_penalty='robin', beta=beta, alpha = alpha, a = a)
    if plot:
        y, u, uy = get_solution_slice(sbp, U)
        plt.rcParams.update({'font.size': 35})
        fig = plt.figure(figsize=(11,9))
        log_line = lambda y: a*np.log(y/beta)
        y_fine, u_fine = read_ins_data(file_name)
        ind = int(len(u_fine))
        plt.semilogx(y_fine[:ind], u_fine[:ind], '--k')
        ind = int(len(u))
        plt.plot(y[:ind], u[:ind])
        y_reg = np.array([y_fine[100] , y_fine[1200]])
        plt.plot(y_reg, log_line(y_reg), '--')
        #plt.plot(y[lw_slice], -0.5*np.ones(y[lw_slice].shape),'xk')
        plt.show()
    return U, sbp

def compute_coarse_grid_error(file_name_fine_solution, U, sbp, lw_slice = None):
    
    phi_slice = slice(1,5)
    y, u, uy = get_solution_slice(sbp, U)
    y_fine, u_fine = read_ins_data(file_name_fine_solution)
    beta, u_tau, a, b = post_process_fine_solution(file_name, e)
    interp = scipy.interpolate.interp1d(y_fine,u_fine)
    u_fine_interp = interp(y)
    u_fine_interp_full = slice_to_full_grid_function(sbp, u_fine_interp)

    err_u = np.abs(U[-1] - u_fine_interp_full)
    err = np.sqrt(sbp.integrate([err_u*err_u]))
    print("err: " + str(err))
    #phi = err_u[1]
    #phi = np.zeros(y.shape)
    #phi[1:] += np.abs(uy[1,1:] - a/y[1:])
    phi = compute_low_error_robin(sbp, U[-1], beta)[0]
    print("phi: " + str(phi[phi_slice]))
    return y, u

def run_simulation_error_in_log_layer(e, lw_slice, alpha = 1, plot = False, file_name = "fine_solution.dat"):
    beta, u_tau, a, b = post_process_fine_solution(file_name, e)

    N_vec = np.array([11])
    y_min = []
    err_vec = []
    for N in N_vec:
        sbp,U = steady_state_couette_flow(N, e, acc_y = 2, 
                lw_slice=lw_slice, interior_penalty='dirichlet', beta=beta, alpha = alpha, a = a)

        y_coarse, u_coarse, _ = get_solution_slice(sbp, U)
        y_fine, u_fine = read_ins_data(file_name)

        interp = scipy.interpolate.interp1d(y_fine,u_fine)
        u_fine_interp = interp(y_coarse)

        err = 0
        for (idx, yi) in enumerate(y_coarse):
            if yi < 1 and yi > 0.1:
                err = np.maximum(err,np.abs(u_coarse[idx] - u_fine_interp[idx]))
        y_min.append(y_coarse[1]-y_coarse[0])
        err_vec.append(err)
    #plt.plot(y_min, err_vec, '*')
    #plt.show()
    #pdb.set_trace()
    print("min delta y: " + str(y_coarse[1]-y_coarse[0]))
    print("max error in log layer: " + str(err))
    return y_coarse, u_coarse

    
if __name__ == '__main__':
    e = 1/10000
    file_name = "data_files/fine_solution_wall_dirichlet.dat"
    y_fine, u_fine = read_ins_data(file_name)
    lw_slice = slice(3,5)
    y, u_mpt = run_simulation_error_in_log_layer(e, lw_slice, file_name = file_name)
    
    #N = 8
    #U, sbp = couette_no_mpt(N, e, file_name = file_name, plot = False)
    #y, u_no_mpt = compute_coarse_grid_error(file_name, U, sbp)
    #U, sbp = couette_mpt(N, e, lw_slice, file_name = file_name, plot = False)
    #y, u_mpt = compute_coarse_grid_error(file_name, U, sbp, lw_slice = lw_slice)
    #y_fine, u_fine = read_ins_data(file_name)
    #print("delta y: " + str(y[1] - y[0]))
    #plt.plot(y,u_no_mpt,'or', label='no mpt')
    plt.semilogx(y_fine, u_fine,'--k',label='fine')
    plt.semilogx(y,u_mpt, '-*',label='mpt')
    plt.plot(y[lw_slice], np.zeros(y[lw_slice].shape), 'xk')
    #beta, u_tau, a, b = post_process_fine_solution(file_name, e)

    #data = a*np.log(y_fine[1:]/beta)
    #plt.plot(y_fine[1:], data, label='log line')
    plt.legend()
    plt.show()
    #write_ins_data("data_files/coarse_solution_sbp21_N21.dat", y, u_no_mpt)

