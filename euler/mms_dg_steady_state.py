import pdb
import numpy as np
from sbpy.grid2d import MultiblockGrid, MultiblockDGSBP
from sbpy.utils import create_convergence_table, solution_to_file
from sbpy.meshes import get_dg_grid, set_bd_info
from euler import solve_euler_ibvp, solve_euler_steady_state
import mms
from sbpy.grid2d import collocate_corners

N_vec   = np.array([2,4,8,12,16])
err_vec = []
order   = 2
width   = 1

for i in range(len(N_vec)):
    N      = N_vec[i]
    h      = width/N
    blocks = get_dg_grid(N,N,h,order)
    collocate_corners(blocks)
    grid   = MultiblockGrid(blocks)
    sbp    = MultiblockDGSBP(grid)

    bd_info = {
            "bd1_x": np.linspace(0,width,N+1),
            "bd1_y": {0},
            "bd2_x": {width},
            "bd2_y": np.linspace(0,width,N+1),
            "bd3_x": np.linspace(0,width,N+1),
            "bd3_y": {width},
            "bd4_x": {0},
            "bd4_y": np.linspace(0,width,N+1)
            }
    #grid.plot_domain(boundary_indices=True)

    boundary_condition = {
            "bd1" : "inflow",
            "bd2" : "pressure",
            "bd4" : "inflow",
            "bd3" : "pressure"
            }

    set_bd_info(grid, bd_info, boundary_condition)
    initu = []
    initv = []
    initp = []
    for (X,Y) in blocks:
        initu.append(np.array(np.ones(X.shape)))
        initv.append(np.array(np.ones(X.shape)))
        initp.append(np.array(np.ones(X.shape)))

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

    U,V,P = solve_euler_steady_state(sbp, boundary_condition, bd_data, initu, initv, initp, force)

    u_analytic = []
    v_analytic = []
    p_analytic = []
    for (X,Y) in blocks:
        u_analytic.append(mms.u(0,X,Y))
        v_analytic.append(mms.v(0,X,Y))
        p_analytic.append(mms.p(0,X,Y))
    
    err = np.sqrt(sbp.integrate((U[-1] - u_analytic)**2) \
                  + sbp.integrate((V[-1] - v_analytic)**2) \
                  + sbp.integrate((P[-1] - p_analytic)**2))

    err_vec.append(err)
    print(err_vec)

create_convergence_table(N_vec, err_vec, 1/N_vec)
solution_to_file(grid,U,V,P,'mms_test/mms_test')
