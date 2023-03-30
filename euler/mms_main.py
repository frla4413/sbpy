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
                   wall_model_as_bc_operator,
                   solve, solve_steady_state, wall_operator,
                   solve_steady_state_newton_krylov,
                   interior_penalty_robin_operator,
                   interior_penalty_dirichlet_operator,
                   interior_penalty_neumann_operator)

def compute_low_error(sbp, u, y0):

    blocks = sbp.grid.get_blocks()
    _,Y = blocks[0]
    uy = sbp.diffy([u])
    K = Y*np.log(Y/y0)
    low_error = np.abs(u - K*uy)
    return low_error[0]

def mms_non_periodic(acc = 2):
    # need import mms_file as mms on top  to run
    # remember: epsilon in mms-file and here need to agree!

    N_vec = np.array([11,21,31,41])
    errors = []
    e = mms.e
    for N in N_vec:
        (X,Y) = np.meshgrid(np.linspace(0,1,N), np.linspace(1,2,N))
        X     = np.transpose(X)
        Y     = np.transpose(Y)

        grid = MultiblockGrid([(X,Y)])
        sbp  = MultiblockSBP(grid, accuracy_x = acc, accuracy_y = acc)

        initu = mms.u(0,X,Y)
        initv = mms.v(0,X,Y)
        initp = mms.p(0,X,Y)
        #initu = np.ones(X.shape)
        #initv = np.zeros(X.shape)
        #initp = np.zeros(X.shape)
        
        beta = mms.beta
        ##K = Y.flatten()*np.log(Y.flatten()/beta)
        #Kinv = 1/K
        #K    = scipy.sparse.diags(K)
        #Kinv = scipy.sparse.diags(Kinv)
        def spatial_op(state):
            t = 0
            S,J,ux,uy,vx,vy = euler_operator(sbp, state, e)

                      #wall_model_as_bc_operator(sbp, state, 0, 's', e, K, Kinv) +\
            Sbd,Jbd = inflow_operator(sbp,state,0,'w',\
                                mms.wn_data(sbp,0,'w', t), \
                                mms.wt_data(sbp,0,'w',t), e) + \
                      pressure_operator(sbp, state, 0, 'e', ux, uy, vx, vy,\
                                mms.normal_outflow_data(sbp,0,'e',t), \
                                mms.tangential_outflow_data(sbp,0,'e',t), e) + \
                      wall_operator(sbp,state,0,'s', e) + \
                      pressure_operator(sbp, state, 0, 'n', ux, uy, vx, vy,\
                                mms.normal_outflow_data(sbp,0,'n',t), \
                                mms.tangential_outflow_data(sbp,0,'n',t), e) 
            S     -= force_operator(sbp,mms.force1,mms.force2,mms.force3,t)[0]
            return S+Sbd, J+Jbd
        U,V,P = solve_steady_state(grid, spatial_op, initu, initv, initp)
        err_u   = np.abs(U[-1] - mms.u(0,X,Y))
        err_v   = np.abs(V[-1] - mms.v(0,X,Y))
        err_p   = np.abs(P[-1] - mms.p(0,X,Y))
        errors.append(np.sqrt(sbp.integrate([err_u*err_u]) +
                              sbp.integrate([err_v*err_v]) + 
                              sbp.integrate([err_p*err_p])))

    create_convergence_table(N_vec, errors, 1/(N_vec - 1))
    #pdb.set_trace()
    return grid,U,V,P


def steady_state_mms_periodic():
    # need import mms_file as mms on top  to run
    # remember: epsilon in mms-file and here need to agree!

    N_vec = np.array([41])
    errors = []
    e = mms.e
    acc = 2
    for N in N_vec:
#        (X,Y) = np.meshgrid(np.linspace(0,1,N), np.linspace(1,2,N))
        y = np.linspace(0,np.pi,N)
        y = -np.cos(y) + 1
        (X,Y) = np.meshgrid(np.linspace(0,1,N), y)
        X     = np.transpose(X)
        Y     = np.transpose(Y)
        X     = X[:-1]
        Y     = Y[:-1]

        grid = MultiblockGrid([(X,Y)])
        sbp  = MultiblockSBP(grid, accuracy_y = acc, periodic = True)
        grid.plot_grid()

        initu = np.array([np.ones(X.shape)])
        initv = np.array([np.ones(X.shape)])
        initp = np.array([np.zeros(X.shape)])

        def spatial_op(state):
            t = 0
            S,J,ux,uy,vx,vy = euler_operator(sbp, state, e)

            Sbd,Jbd = inflow_operator(sbp,state,0,'s',\
                                mms.wn_data(sbp,0,'s', t), \
                                mms.wt_data(sbp,0,'s',t), e) + \
                      pressure_operator(sbp, state, 0, 'n', ux, uy, vx, vy,\
                                mms.normal_outflow_data(sbp,0,'n',t), \
                                mms.tangential_outflow_data(sbp,0,'n',t), e) 

            S     -= force_operator(sbp,mms.force1,mms.force2,mms.force3,t)[0]

            return S+Sbd, J+Jbd
        U,V,P = solve_steady_state(grid, spatial_op, initu, initv,initp)

        err_u   = np.abs(U[-1] - mms.u(0,X,Y))
        err_v   = np.abs(V[-1] - mms.v(0,X,Y))
        err_p   = np.abs(P[-1] - mms.p(0,X,Y))
        errors.append(np.sqrt(sbp.integrate([err_u*err_u]) +
                              sbp.integrate([err_v*err_v]) + 
                              sbp.integrate([err_p*err_p])))

    create_convergence_table(N_vec, errors, 1/(N_vec - 1))
    #pdb.set_trace()
    return grid,U,V,P


if __name__ == '__main__':
    grid,U,V,P = mms_non_periodic(acc = 2)
    #grid,U,V,P = steady_state_mms_periodic()
    #export_to_tecplot(grid,U[-1],V[-1],P[-1],"data_files/mms_time_indep_N11_sbp21_one_pen_robin.dat")
