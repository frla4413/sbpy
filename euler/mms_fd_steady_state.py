import pdb
import numpy as np
from sbpy.grid2d import MultiblockGrid, MultiblockSBP, collocate_corners
from sbpy.utils import create_convergence_table, solution_to_file
from sbpy.meshes import get_bump_grid, set_bd_info
from euler import solve_euler_steady_state
import mms
from sbpy.grid2d import load_p3d

N_vec   = np.array([41,51])
err_vec = []
acc     = 4

for i in range(len(N_vec)):
    N = N_vec[i]

    X0,Y0 = get_bump_grid(N)
    X1 = X0 
    Y1 = Y0 + 0.2
    Y2 = Y1 + 0.2

    blocks = np.array([(X0,Y0), (X1,Y1)])#, (X1,Y2)])
    collocate_corners(blocks)
    grid   = MultiblockGrid(blocks)
    sbp    = MultiblockSBP(grid, accuracy=acc)

    num_blocks = len(blocks)
    y0 = lambda x: 0.0625*np.exp(-25*x**2)
    
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

#    grid.plot_domain(boundary_indices=True)
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


    bd_data = {
        "wn_data": mms.wn_data,
        "wt_data": mms.wt_data,
        "p_data" : mms.p_data
        }

    force = {
        "force1": mms.force1,
        "force2": mms.force2,
        "force3": mms.force3
        }

    U,V,P = solve_euler_steady_state(sbp, boundary_condition, bd_data,\
                                       initu, initv, initp, force)

    u_analytic = []
    v_analytic = []
    p_analytic = []
    for (X,Y) in blocks:
        u_analytic.append(mms.u(0,X,Y))
        v_analytic.append(mms.v(0,X,Y))
        p_analytic.append(mms.p(0,X,Y))
    
    err = np.sqrt(sbp.integrate((U[-1] - u_analytic)**2) \
                  + sbp.integrate((V[-1] - v_analytic)**2) \
                  + sbp.integrate((P[-1] - p_analytic)**2) )

    err_vec.append(err)
    print(err_vec)

create_convergence_table(N_vec, err_vec, 1/N_vec)
solution_to_file(grid,U,V,P,'mms_test/mms_test')
