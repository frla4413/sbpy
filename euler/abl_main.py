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
                   outflow_operator, pressure_operator,
                   solve, solve_steady_state, wall_operator,
                   periodic_in_x_operator, solve_steady_state_newton_krylov)

from sbpy.abl_utils import evaluate_1d_in_y_function_on_grid, bd_func_damping, bd_func_no_damping, read_ins_data, correct_bd_func_damping, get_plus_variables

def steady_state_couette_flow(N, e = 0.01, acc_y = 2, interior_penalty = False, 
                              lw_slice = slice(None,None), beta = 0):
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

        Sbd,Jbd = inflow_operator(sbp,state,0,'n', 0, -speed, e, turb_model = turbulence, 
                                   lm_func = lm_bd_func) + \
                  inflow_operator(sbp,state,0,'s', 0, -speed, e, turb_model = turbulence, 
                                   lm_func = lm_bd_func)
        return S+Sbd, J+Jbd

    U,V,P = solve_steady_state(grid, spatial_op, initu, initv, initp)
    return sbp,U

def one_run(N = 100, e = 0.01):

    sbp,U = steady_state_couette_flow(N, e, acc_y = 4)
    blocks = sbp.grid.get_blocks()
    Y = blocks[0][1]
   
    u = U[-1][2]
    y = Y[1]
    
    uy = sbp.diffy([U[-1]])
    uy = uy[0][:,0]
    uy = np.mean(uy)
    tau_w = e*uy
    u_tau = np.sqrt(tau_w)
    Re_tau = u_tau/e
    print("Re_tau: " + str(Re_tau), "u_tau: ", str(u_tau))
    plt.rcParams.update({'font.size': 35})
    fig = plt.figure(figsize=(11,9))
    fig2 = plt.figure(figsize=(11,9))
    ax  = fig.gca()
    ax.plot(y, u/np.max(u))
    y_dns,u_dns = read_ins_data("um.dat")
    ax.plot(y_dns+1, u_dns,'--k')

    #log-plot
    ind = int(len(u)/2)
    y_dns_plus,u_dns_plus = read_ins_data("um_plus.dat")
    #u_tau = 0.05
    y_plus,u_plus = get_plus_variables(y[0:ind], u[0:ind], u_tau, e)
    ax2 = fig2.gca()
    ax2.semilogx(y_plus,u_plus)
    ax2.semilogx(y_dns_plus,u_dns_plus,'--k')
    kappa = 0.41
    log_line = lambda y_plus: np.log(y_plus)/kappa + 5.2
    y_vec = np.array([y_dns_plus[-80], y_dns_plus[-10]])
    ax2.plot(y_vec, log_line(y_vec))
    plt.show()

if __name__ == '__main__':
    one_run(N = 1000, e = 1/10000)
