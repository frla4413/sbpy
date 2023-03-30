import pdb

import matplotlib.pyplot as plt
import numpy as np
import scipy
from sbpy.grid2d import MultiblockGrid, MultiblockSBP
from sbpy.utils import (create_convergence_table, export_to_tecplot, surf_plot,
                       get_gauss_initial_data)

#import mms_kovaszany as mms
import mms as mms

from euler import (euler_operator, force_operator, inflow_operator,
                   outflow_operator, pressure_operator,
                   solve, solve_steady_state, wall_operator,
                   solve_steady_state_newton_krylov,
                   interior_penalty_dirichlet_operator, 
                   interior_penalty_neumann_operator,
                   interior_penalty_robin_operator)

from sbpy.utils import (create_convergence_table)

def steady_state_mms_periodic(N, interior_penalty = False):
    # need import mms_file as mms on top  to run
    # remember: epsilon in mms-file and here need to agree!

    e = mms.e
    (X,Y) = np.meshgrid(np.linspace(0,1,N), np.linspace(1,2,N))
    X     = np.transpose(X)
    Y     = np.transpose(Y)
    X     = X[:-1]
    Y     = Y[:-1]
    acc   = 2

    grid = MultiblockGrid([(X,Y)])
    sbp  = MultiblockSBP(grid, accuracy_y = acc, periodic = True)

    initu = np.array([np.ones(X.shape)])
    initv = np.array([np.ones(X.shape)])
    initp = np.array([np.zeros(X.shape)])
    initu = mms.u(0,X,Y)
    initv = mms.v(0,X,Y)
    initp = mms.p(0,X,Y)

    data = mms.u(0,X,Y)
    lw_slice1 = slice(None,None)
    lw_slice2 = slice(None,None)
    force = force_operator(sbp,mms.force1,mms.force2,mms.force3,0)[0]
    def spatial_op(state):
        t = 0
        S,J,ux,uy,vx,vy = euler_operator(sbp, state, e)

        Sbd,Jbd = wall_operator(sbp,state,0,'s', e) +\
                  pressure_operator(sbp, state, 0, 'n', ux, uy, vx, vy,\
                            mms.normal_outflow_data(sbp,0,'n',t), \
                            mms.tangential_outflow_data(sbp,0,'n',t), e) 
        S     -= force_operator(sbp,mms.force1,mms.force2,mms.force3,t)[0]
        return S+Sbd, J+Jbd
    U,V,P = solve_steady_state(grid, spatial_op, initu, initv,initp)
    return grid,U,V,P

def steady_state_mms(N, interior_penalty = False, lw_slice1 = None, lw_slice2 = None):
    # need import mms_file as mms on top  to run
    # remember: epsilon in mms-file and here need to agree!

    e = mms.e
    beta = mms.beta
    (X,Y) = np.meshgrid(np.linspace(0,1,N), np.linspace(1,2,N))
    X     = np.transpose(X)
    Y     = np.transpose(Y)
    acc   = 2

    grid = MultiblockGrid([(X,Y)])
    sbp  = MultiblockSBP(grid, accuracy_y = acc)

    initu = np.array([np.ones(X.shape)])
    initv = np.array([np.ones(X.shape)])
    initp = np.array([np.zeros(X.shape)])

    force = force_operator(sbp,mms.force1,mms.force2,mms.force3,0)[0]
    def spatial_op(state):
        t = 0
        S,J,ux,uy,vx,vy = euler_operator(sbp, state, e)

        Sbd,Jbd = inflow_operator(sbp,state,0,'w',\
                            mms.wn_data(sbp,0,'w', t), \
                            mms.wt_data(sbp,0,'w',t), e) + \
                  wall_operator(sbp,state,0,'s', e) +\
                  pressure_operator(sbp, state, 0, 'e', ux, uy, vx, vy,\
                            mms.normal_outflow_data(sbp,0,'e',t), \
                            mms.tangential_outflow_data(sbp,0,'e',t), e) + \
                  pressure_operator(sbp, state, 0, 'n', ux, uy, vx, vy,\
                            mms.normal_outflow_data(sbp,0,'n',t), \
                            mms.tangential_outflow_data(sbp,0,'n',t), e) 
        S     -= force_operator(sbp,mms.force1,mms.force2,mms.force3,t)[0]

        if interior_penalty == 'dirichlet':
            data = mms.u(0,X,Y)
            Si, Ji = interior_penalty_dirichlet_operator(sbp, state, 0, lw_slice1, 
                     lw_slice2, data)
            S += Si
            J += Ji

        if interior_penalty == 'neumann':
            data = mms.u_y(0,X,Y)
            Si, Ji = interior_penalty_neumann_operator(sbp, state, 0, lw_slice1, 
                     lw_slice2, data)
            S += Si
            J += Ji

        if interior_penalty == 'robin':
            Si, Ji = interior_penalty_robin_operator(sbp, state, 0, lw_slice1, lw_slice2, uy, e, beta)
            S += Si
            J += Ji
        return S+Sbd, J+Jbd

    
    #U,V,P = solve_steady_state(grid, spatial_op, initu, initv,initp)
    U,V,P = solve_steady_state_newton_krylov(grid, spatial_op, initu, initv,initp)
    err_u = np.abs(U[-1] - mms.u(0,X,Y))
    err_tot = np.sqrt(sbp.integrate([err_u*err_u]))

    #indicator = err_u
    uy = sbp.diffy([U[-1]])[0]
    #indicator += np.abs(uy - mms.u_y(0,X,Y))

    K    = Y*np.log(Y/beta)
    indicator = np.abs(U[-1] - K*uy)
    ind_array = np.flip(indicator.argsort(axis=None))[0:1]
    ind_max = np.unravel_index(ind_array, indicator.shape)

    print("l2-error: " + str(err_tot))
    print("indicator[lw_slice1,lw_slice2] :" + str(indicator[lw_slice1,lw_slice2]))
    print("ind_max: " + str(ind_max[0][0]) + " " + str(ind_max[1][0]))
    print("indicator[ind_max]: " + str(indicator[ind_max[0][0], ind_max[1][0]]))
    slice1 = np.array([10,10])
    slice2 = np.array([3,2])
    print("indicator[8,10,  1,2]: " + str(indicator[slice1,slice2]))
    grid.plot_grid_function(indicator)
    return grid,U,V,P

if __name__ == '__main__':
    grid, U,V,P = steady_state_mms(N=11, lw_slice1= np.array([10]), lw_slice2=np.array([3]), interior_penalty='robin')
