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
                   interior_penalty_neumann_operator, interior_penalty_robin_operator,
                   wall_model_as_bc_operator)

from sbpy.abl_utils import (evaluate_1d_in_y_function_on_grid, bd_func_damping, 
                            bd_func_no_damping, read_ins_data, get_plus_variables, 
                            compute_u_tau, write_ins_data, get_solution_slice,
                            compute_low_error_slice, compute_uy_lw_error_slice,
                            post_process_fine_solution, slice_to_full_grid_function, 
                            compute_low_error_robin)

def steady_state_couette_flow(N, e = 0.01, acc_y = 2, interior_penalty = False,
                              lw_slice = slice(None,None), beta = 0, alpha = 1,
                              a = 0, u_tau = 1):
    Nx = 4
    middle = 1
    (X,Y) = np.meshgrid(np.linspace(0,1,Nx), np.linspace(0.2,2,N))
    X     = np.transpose(X)
    Y     = np.transpose(Y)
    X     = X[:-1]
    Y     = Y[:-1]
    y = Y[0]
    #y = np.linspace(0.25,1,N)   # --> for c = 2
    y = np.linspace(0.31,1,N)   # --> for c = 3
    c = 3
    b1 = y**c
    b2 = (1 - y)**c
    y = b1/(b1 + b2)
    y = 2*y
    #y = np.linspace(0.3,2,N)

    (X,Y) = np.meshgrid(np.linspace(0,1,Nx), y)
    X     = np.transpose(X)
    Y     = np.transpose(Y)
    X     = X[:-1]
    Y     = Y[:-1]
    
    #damping function for turbulence
    bd_func_1d = lambda y: bd_func_damping(y, e, u_tau, middle)
    lm_bd_func = lambda y: evaluate_1d_in_y_function_on_grid(y, bd_func_1d)
 
    grid = MultiblockGrid([(X,Y)])
    sbp  = MultiblockSBP(grid, accuracy_x = 2, accuracy_y = acc_y, periodic = True)
    #grid.plot_grid()

    speed = 2
    initu = speed*np.array([np.ones(X.shape)])
    initu = np.array([np.zeros(X.shape)])
    initv = np.array([np.zeros(X.shape)])
    initp = np.array([np.zeros(X.shape)])
    dissipation = False
    art_diss_matrix = generate_artificial_dissipation_matrix_2D(grid, alpha = 0.1)
    art_diss_matrix = scipy.sparse.kron(scipy.sparse.eye(3), art_diss_matrix)
    K    = Y.flatten()*np.log(Y.flatten()/beta)
    Kinv = 1/K
    K    = scipy.sparse.diags(K)
    Kinv = scipy.sparse.diags(Kinv)

    def spatial_op(state):
        turbulence = True
        data1 = a*np.log(y[0]/beta)
        S,J,ux,uy,vx,vy = euler_operator(sbp, state, e, turb_model = turbulence, 
                                         lm_func = lm_bd_func)
        Sbd,Jbd = inflow_operator(sbp,state, 0, 'n', 0, -speed, e, turb_model = turbulence, 
                                   lm_func = lm_bd_func) +\
                  wall_model_as_bc_operator(sbp, state,0,'s', e, K, Kinv, turb_model=turbulence,
                                            lm_func=lm_bd_func)
                  #wall_operator(sbp, state, 0, 's', e, turb_model=turbulence, lm_func = lm_bd_func)
                  #inflow_operator(sbp,state, 0, 's', 0, 0, e, turb_model = turbulence, 
                  #                 lm_func = lm_bd_func)

        if interior_penalty == 'robin':
            lw_slice1 = slice(None,None)
            lw_slice2 = lw_slice
            data = np.zeros(Y.shape)
            Si, Ji = interior_penalty_robin_operator(sbp, state, 0, lw_slice1, lw_slice2, uy, e, beta)
            S += Si
            J += Ji

        if interior_penalty == 'dirichlet':
            lw_slice1 = slice(None,None)
            lw_slice2 = lw_slice
            data = a*np.log(Y/beta)
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


def coarse_simulation(N = 101, e = 1/10000, alpha = 1, file_name = "fine_solution.dat", lw_slice = None,
        penalty_type = None):
    beta, u_tau, a, b = post_process_fine_solution(file_name, e)

    # mpt-simulation
    sbp,U = steady_state_couette_flow(N, e, acc_y = 2, alpha = alpha, 
            beta = beta, u_tau = u_tau, lw_slice = lw_slice, a = a, interior_penalty=penalty_type)

    u_mpt = U[-1][2]
    blocks = sbp.grid.get_blocks()
    Y = blocks[0][1]
    y = Y[1]
    return y, u_mpt, sbp, U

def run_simulations(N = 101, e = 1/10000, alpha = 1, file_name = "fine_solution.dat",lw_slice = None, penalty_type = None):


    indicator_slice = slice(0,4)
    y_fine, u_fine = read_ins_data(file_name)
    beta, u_tau, a, b = post_process_fine_solution(file_name, e)

    # No mpt 
    y_no_mpt, u_no_mpt, sbp, U = coarse_simulation(N = N, e = e, alpha = alpha, file_name=file_name)
    
    interp = scipy.interpolate.interp1d(y_fine,u_fine)
    u_fine_interp = interp(y_no_mpt)
    u_fine_interp_full = slice_to_full_grid_function(sbp, u_fine_interp)
   
    err_u = np.abs(U[-1] - u_fine_interp_full)
    err_no_mpt = np.sqrt(sbp.integrate([err_u*err_u])) 

    #indicator_no_mpt = err_u[2]
    #uy = sbp.diffy([U[-1]])[0][2]
    #indicator_no_mpt += np.abs(uy - a/y_no_mpt)
    #indicator_no_mpt = indicator_no_mpt[indicator_slice]
    #indicator_no_mpt = compute_low_error_robin(sbp, U[-1], beta)[0][indicator_slice]
                                                                                                    
    # mpt
    y_mpt, u_mpt, sbp, U = coarse_simulation(N = N, e = e, alpha = alpha, 
                        file_name=file_name, lw_slice=lw_slice, 
                        penalty_type=penalty_type)
    err_u = np.abs(U[-1] - u_fine_interp_full)
    err_mpt = np.sqrt(sbp.integrate([err_u*err_u])) 

    #indicator_mpt = err_u[2]
    #uy = sbp.diffy([U[-1]])[0][2]
    #indicator_mpt += np.abs(uy - a/y_no_mpt)
    #indicator_mpt = indicator_mpt[indicator_slice]
    #indicator_mpt = compute_low_error_robin(sbp, U[-1], beta)[0][indicator_slice]

    #print("Indicator, no mpt: " + str(indicator_no_mpt))
    #print("Indicator, mpt: " + str(indicator_mpt))


    print("Err no mpt: " + str(err_no_mpt))
    print("Err mpt: " + str(err_mpt))

    #plt.plot(y_mpt, u_mpt,'or', label='mpt')
    #plt.plot(y_no_mpt, u_no_mpt,'*',linewidth=3, label='No mpt')
    #plt.plot(y_fine, u_fine,'--k',label='fine')
    #plt.plot(y_mpt[lw_slice], np.zeros(y_mpt[lw_slice].shape),'xk', label="mpt pos.")
    #plt.legend()
    #plt.show()

    plt.semilogx(y_mpt, u_mpt,'*', label='mpt')
    plt.plot(y_no_mpt, u_no_mpt,'*',linewidth=3, label='No mpt')
    plt.plot(y_fine, u_fine,'--k',label='fine')
    #plt.plot(y_fine, y_fine*u_tau*u_tau/e)
    plt.plot(y_mpt[lw_slice], np.zeros(y_mpt[lw_slice].shape),'xk')
    plt.legend()
    plt.show()

    #err_no_mpt = 0
    #err_mpt = 0
    #for (idx, yi) in enumerate(y_no_mpt):
    #    if yi < 1 and yi > 0.1:
    #        err_no_mpt = np.maximum(err_no_mpt,
    #                                np.abs(u_no_mpt[idx] - u_fine_interp[idx]))
    #        err_mpt = np.maximum(err_mpt,np.abs(u_mpt[idx] - u_fine_interp[idx]))
    #print("min delta y: " + str(y_no_mpt[1]-y_no_mpt[0]))
    #print("max error in log layer, no mpt: " + str(err_no_mpt))
    #print("max error in log layer, mpt: " + str(err_mpt))


if __name__ == '__main__':
    e = 1/10000
    file_name = "data_files/fine_solution_wall_dirichlet.dat"
    lw_slice = slice(1,2)
    run_simulations(N = 8, e = e, alpha = 1, file_name = file_name, lw_slice=lw_slice, 
                    penalty_type = 'robin')
    #print(y[1] - y[0])
    #write_ins_data("data_files/wall_model_bc_N8_two_mpt_slice02.dat", y, u)
    #beta, u_tau, a, b = post_process_fine_solution(file_name, e)
    #pdb.set_trace()
