import pdb
import numpy as np
from sbpy.grid2d import MultiblockGrid, MultiblockSBP, collocate_corners
from sbpy.utils import create_convergence_table, solution_to_file
from sbpy.meshes import get_bump_grid, set_bd_info
from euler import solve_euler_ibvp
import mms
from sbpy.grid2d import load_p3d

N_vec         = np.array([31])
acc           = 4
dt            = 1/200
num_timesteps = 2000
err_vec       = []

for i in range(len(N_vec)):
    N = N_vec[i]

    X0,Y0 = get_bump_grid(N)
    X1 = X0 
    Y1 = Y0 + 0.2
    Y2 = Y1 + 0.2
    blocks = np.array([(X0,Y0),(X1,Y1)])#, (X1,Y2)])
    collocate_corners(blocks)
    grid   = MultiblockGrid(blocks)
    sbp    = MultiblockSBP(grid, accuracy=acc)

    y0 = lambda x: 0.0625*np.exp(-25*x**2)
    num_blocks = len(blocks)
    bd_info = {
            "bd1_x": [-0.5, 0.5],
            "bd1_y": [y0(-0.5), y0(0.5)],
            "bd2_x": [0.5],
            "bd2_y": [y0(0.5) +i*0.2 for i in range(num_blocks+1)],
            "bd3_x": [-0.5,0.5],
            "bd3_y": [y0(-0.5) + num_blocks*0.2, y0(0.5) + num_blocks*0.2],
            "bd4_x": [-0.5],
            "bd4_y": [y0(0.5) +i*0.2 for i in range(num_blocks+1)]
            }

    #bd_info = {
    #        "bd1_x": np.linspace(-0.5,0.5,N),
    #        "bd1_y": y0(np.linspace(-0.5,0.5,N)),
    #        "bd2_x": {0.5},
    #        "bd2_y": [y0(0.5) +i*0.2 for i in range(num_blocks+1)],
    #        "bd3_x": np.linspace(-0.5,0.5,N),
    #        "bd3_y": num_blocks*0.2 + y0(np.linspace(-0.5,0.5,N)),
    #        "bd4_x": {-0.5},
    #        "bd4_y": [y0(0.5) +i*0.2 for i in range(num_blocks+1)]
    #        }

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
        initu.append(mms.u(0,X,Y))
        initv.append(mms.v(0,X,Y))
        initp.append(mms.p(0,X,Y))


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

    #U,V,P = solve_euler_ibvp(sbp, boundary_condition, bd_data, initu, initv, initp, \
    #                         dt, num_timesteps, force)

    name = 'plots/sol'
    U,V,P = solve_euler_ibvp(sbp, boundary_condition, bd_data, initu, initv, initp,\
                         dt, num_timesteps, name_base = name)

    t_end      = dt*num_timesteps
    print(t_end)
    u_analytic = []
    v_analytic = []
    p_analytic = []
    for (X,Y) in blocks:
        u_analytic.append(mms.u(t_end,X,Y))
        v_analytic.append(mms.v(t_end,X,Y))
        p_analytic.append(mms.p(t_end,X,Y))
    
    err = np.sqrt(sbp.integrate((U - u_analytic)**2) \
                  + sbp.integrate((V - v_analytic)**2) \
                  + sbp.integrate((P - p_analytic)**2) )

    err_vec.append(err)
    print(err_vec)

create_convergence_table(N_vec, err_vec, 1/(N_vec-1))
#solution_to_file(grid,U,V,P,'mms_const/mms_const')
