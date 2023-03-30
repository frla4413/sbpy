import pdb
import numpy as np
from sbpy.grid2d import MultiblockGrid, MultiblockSBP
from sbpy.utils import create_convergence_table, solution_to_file
from sbpy.meshes import get_bump_grid
from euler import solve_euler_ibvp, solve_euler_steady_state
import mms
from sbpy.grid2d import load_p3d

N_vec   = np.array([21,41])
err_vec = []

dt            = 1e-2
num_timesteps = 10
acc           = 4

for i in range(len(N_vec)):
    N = N_vec[i]

    X0,Y0 = get_bump_grid(N)
    X1 = X0 
    Y1 = Y0 + 0.8
    Y2 = Y1 + 0.8

    indices = {
        "inflow_idx"  : {0,2,4,7},
        "pressure_idx": {1,3,5,6},
        "outflow_idx" : {},
        "wall_idx"    : {}
        }

    blocks = np.array([(X0,Y0), (X1,Y1), (X1,Y2)])
    grid   = MultiblockGrid(blocks)
    sbp    = MultiblockSBP(grid, accuracy=acc)

    #grid.plot_domain(boundary_indices=True)
    initu = []
    initv = []
    initp = []
    for (X,Y) in blocks:
        initu.append(np.array([mms.u(0,X,Y)]))
        initv.append(np.array([mms.v(0,X,Y)]))
        initp.append(np.array([mms.p(0,X,Y)]))
        #initu.append(np.array(np.ones(X.shape)))
        #initv.append(np.array(np.ones(X.shape)))
        #initp.append(np.array(np.ones(X.shape)))



    def wn_data(sbp, block_idx, side,t):
        n        = sbp.get_normals(block_idx,side)
        nx       = n[:,0] 
        ny       = n[:,1] 
        xbd, ybd = grid.get_boundary(block_idx,side)
        return mms.u(t,xbd,ybd)*nx + mms.v(t,xbd,ybd)*ny

    def wt_data(sbp, block_idx, side,t):
        n        = sbp.get_normals(block_idx,side)
        nx       = n[:,0] 
        ny       = n[:,1] 
        xbd, ybd = grid.get_boundary(block_idx,side)
        return -mms.u(t,xbd,ybd)*ny + mms.v(t,xbd,ybd)*nx

    def p_data(sbp,block_idx,side,t):
        xbd, ybd = grid.get_boundary(block_idx,side)
        return mms.p(t,xbd,ybd)

    bd_data = {
        "wn_data": wn_data,
        "wt_data": wt_data,
        "p_data" : p_data
        }

    force = {
        "force1": mms.force1,
        "force2": mms.force2,
        "force3": mms.force3
        }

    #U,V,P = solve_euler(sbp, indices, bd_data, initu, initv, initp, \
    #                    dt, num_timesteps, force)
    U,V,P = solve_euler_steady_state(sbp, indices, bd_data, \
                                    initu, initv, initp, force)

    t_end      = dt*num_timesteps
    u_analytic = []
    v_analytic = []
    p_analytic = []
    for (X,Y) in blocks:
        u_analytic.append(mms.u(t_end,X,Y))
        v_analytic.append(mms.v(t_end,X,Y))
        p_analytic.append(mms.p(t_end,X,Y))
    
    err = np.sqrt(sbp.integrate((U[-1] - u_analytic)**2) \
                  + sbp.integrate((V[-1] - v_analytic)**2) \
                  + sbp.integrate((P[-1] - p_analytic)**2) )

    err_vec.append(err)
    print(err_vec)

create_convergence_table(N_vec, err_vec, 1/N_vec)
solution_to_file(grid,U,V,P,'mms_test/mms_test')
