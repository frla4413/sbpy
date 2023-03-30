import pdb
import numpy as np
from sbpy.grid2d import MultiblockGrid, MultiblockSBP
from sbpy.utils import create_convergence_table, solution_to_file
from sbpy.meshes import get_annulus_grid, set_bd_info
from euler import solve_euler_ibvp, solve_euler_steady_state
import mms
from sbpy.grid2d import load_p3d

N_vec   = np.array([41,51])
err_vec = []
acc     = 4

for i in range(len(N_vec)):
    N = N_vec[i]

    blocks = get_annulus_grid(N,5)
    grid = MultiblockGrid(blocks)
    sbp = MultiblockSBP(grid, accuracy=acc)

    x_pos      = lambda r, th:  r*np.cos(th)
    y_pos      = lambda r, th:  r*np.sin(th)
    num_blocks = len(blocks)
    jump       = 2*np.pi/num_blocks

    bd_info = {
            "bd1_x": [x_pos(2.0,jump*i) for i in range(2,6)],
            "bd1_y": [y_pos(2.0,jump*i) for i in range(2,6)],
            "bd2_x": [x_pos(2.0,jump*i) for i in range(3)],
            "bd2_y": [y_pos(2.0,jump*i) for i in range(3)],
            "bd3_x": [x_pos(0.1,jump*i) for i in range(3)],
            "bd3_y": [y_pos(0.1,jump*i) for i in range(3)],
            "bd4_x": [x_pos(0.1,jump*i) for i in range(2,6)],
            "bd4_y": [y_pos(0.1,jump*i) for i in range(2,6)]
            }

    #grid.plot_domain(boundary_indices=True)
    boundary_condition = {
            "bd1" : "inflow",
            "bd2" : "pressure",
            "bd3" : "inflow",
            "bd4" : "pressure"
            }

    set_bd_info(grid, bd_info, boundary_condition)

    initu = []
    initv = []
    initp = []
    for (X,Y) in blocks:
        initu.append(mms.u(0,X,Y))
        initv.append(mms.v(0,X,Y))
        initp.append(mms.p(0,X,Y))
        #initu.append(np.array(np.ones(X.shape)))
        #initv.append(np.array(np.ones(X.shape)))
        #initp.append(np.array(np.ones(X.shape)))


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
