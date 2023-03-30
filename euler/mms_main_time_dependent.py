import pdb

import matplotlib.pyplot as plt
import numpy as np
import scipy
from sbpy.grid2d import MultiblockGrid, MultiblockSBP
from sbpy.utils import (create_convergence_table, export_to_tecplot, surf_plot,
                       get_gauss_initial_data)
from sbpy.abl_utils import read_ins_data

#import mms_kovaszany as mms
import mms as mms


from euler import (euler_operator, force_operator, inflow_operator,
                   outflow_operator, pressure_operator, wall_operator,
                   wall_model_as_bc_operator,
                   solve, solve_with_mpt, solve_steady_state, 
                   solve_steady_state_newton_krylov,
                   interior_penalty_robin_operator,
                   interior_penalty_dirichlet_operator)


def time_dependent_mms(acc = 2):
    # need import mms_file as mms on top  to run
    # remember: epsilon in mms-file and here need to agree!

    e = mms.e
    N_vec = np.array([11,21,41,61,81])
    errors = []
    errors_u = []
    errors_v = []
    errors_p = []

    beta = mms.beta
    dt = 1e-5
    num_timesteps = 40

    for N in N_vec:
        (X,Y) = np.meshgrid(np.linspace(0,1,N), np.linspace(1,2,N))
        X     = np.transpose(X)
        Y     = np.transpose(Y)

        grid = MultiblockGrid([(X,Y)])
        sbp  = MultiblockSBP(grid, accuracy_x = acc, accuracy_y = acc)

        initu = mms.u(0,X,Y)
        initv = mms.v(0,X,Y)
        initp = mms.p(0,X,Y)
        
        K    = Y.flatten()*np.log(Y.flatten()/beta)
        Kinv = 1/K
        K    = scipy.sparse.diags(K)
        Kinv = scipy.sparse.diags(Kinv)

        def spatial_op(state, t):
            S,J,ux,uy,vx,vy = euler_operator(sbp, state, e)

                      #wall_model_as_bc_operator(sbp, state, 0, 'n', e, K, Kinv) + \
            Sbd,Jbd = inflow_operator(sbp,state,0,'w',\
                                mms.wn_data(sbp,0,'w', t), \
                                mms.wt_data(sbp,0,'w',t), e) + \
                      pressure_operator(sbp, state, 0, 'e', ux, uy, vx, vy,\
                                mms.normal_outflow_data(sbp,0,'e',t), \
                                mms.tangential_outflow_data(sbp,0,'e',t), e) + \
                      pressure_operator(sbp, state, 0, 'n', ux, uy, vx, vy,\
                                mms.normal_outflow_data(sbp,0,'n',t), \
                                mms.tangential_outflow_data(sbp,0,'n',t), e) +\
                      wall_model_as_bc_operator(sbp, state, 0, 's', e, K, Kinv)
                      
            S     -= force_operator(sbp,mms.force1,mms.force2,mms.force3,t)[0]
            return S+Sbd, J+Jbd

        U,V,P = solve(grid, spatial_op, initu, initv,initp, dt, num_timesteps, sbp)
        t_end = dt*num_timesteps
        err_u = np.abs(U[-1] - mms.u(t_end,X,Y))
        err_v = np.abs(V[-1] - mms.v(t_end,X,Y))
        err_p   = np.abs(P[-1] - mms.p(t_end,X,Y))

        err_tot = np.sqrt(sbp.integrate([err_u*err_u]) +
                          sbp.integrate([err_v*err_v]) + 
                          sbp.integrate([err_p*err_p]))

        errors.append(err_tot)
        errors_u.append(np.sqrt(sbp.integrate([err_u*err_u])))
        errors_v.append(np.sqrt(sbp.integrate([err_v*err_v])))
        errors_p.append(np.sqrt(sbp.integrate([err_p*err_p])))
        foo = (U[-1].flatten() - K*sbp.diffy([U[-1]]).flatten()).reshape(N,N)
    create_convergence_table(N_vec, errors, 1/(N_vec - 1))
    create_convergence_table(N_vec, errors_u, 1/(N_vec - 1))
    create_convergence_table(N_vec, errors_v, 1/(N_vec - 1))
    create_convergence_table(N_vec, errors_p, 1/(N_vec - 1))
    grid.plot_grid_function(U[-1])
    pdb.set_trace()
    return grid,U,V,P


if __name__ == '__main__':
    grid,U,V,P = time_dependent_mms(acc = 4)
