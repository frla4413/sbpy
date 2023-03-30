import pdb

import scipy
import numpy as np
import matplotlib.pyplot as plt

from sbpy.utils import create_convergence_table, export_to_tecplot, surf_plot
from sbpy.grid2d import MultiblockGrid, MultiblockSBP
from euler import euler_operator, wall_operator, outflow_operator, pressure_operator, inflow_operator, solve, solve_steady_state, force_operator, interior_low_operator

import mms_abl as mms

def steady_state_mms_abl(acc = 2):

    # need import mms_file as mms on top  to run
    # remember: epsilon in mms-file and here need to agree!

    e = mms.e
    N = 8

    (X,Y) = np.meshgrid(np.linspace(0,1,N), np.linspace(0,2,N))
    X     = np.transpose(X)
    Y     = np.transpose(Y)

    grid = MultiblockGrid([(X,Y)])
    sbp  = MultiblockSBP(grid, accuracy = acc)

    initu = np.array([4*np.ones(X.shape)])
    initv = np.array([np.ones(X.shape)])
    initp = np.array([np.ones(X.shape)])

    lw_slice1  = np.array((4)) #slice(10,11,None)
    lw_slice2  = np.array((7))  #slice(3,4,None)

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
        S     -= force_operator(sbp,mms.force1,mms.force2,mms.force3,t)[0]
        Si, Ji = interior_low_operator(sbp, state, 0, lw_slice1, lw_slice2, uy, e)

        return S+Sbd, J+Jbd
    U,V,P = solve_steady_state(grid, spatial_op, initu, initv,initp)

    err = np.array([U[-1]-mms.u(0,X,Y), V[-1]-mms.v(0,X,Y), \
                   P[-1] - mms.p(0,X,Y)])

    err       = err.flatten()
    block_idx = 0
    P_big     = scipy.sparse.kron(scipy.sparse.eye(3),sbp.get_full_P(block_idx))
    l2_error  = np.sqrt(np.transpose(err)@P_big@err)

    # post-processing
    slice1  = slice(int(N/2),None,None)
    slice2  = slice(int(N/2),None,None)

    x       = X[slice1,slice2]
    y       = Y[slice1,slice2]
    u       = U[-1][slice1,slice2]
    uy      = sbp.diffy([U[-1]])
    uy      = uy[0,slice1,slice2]
    K       = y*np.log(y/mms.beta)

    # compute LW-error  
    lw_error = np.abs(u - K*uy)
    max_lw   = np.max(lw_error)
    ind_max  = np.where(lw_error == max_lw)

    print("max_ind: "  + str(ind_max))
    print("lw_error: " + str(lw_error[lw_slice1-4,lw_slice2-4]))

    surf_plot(x,y,np.abs(lw_error))
    err_u = np.abs(U[-1] - mms.u(0,X,Y))
    grid.plot_grid_function(err_u)
    Z = np.zeros(X.shape)
    Z[lw_slice1,lw_slice2] = 1
    surf_plot(X,Y,Z)
    
    print("L2-error: " + str(l2_error))
    return grid,U,V,P


if __name__ == '__main__':

    grid,U,V,P = steady_state_mms_abl(acc = 2)
    grid.plot_grid_function(U[-1])
