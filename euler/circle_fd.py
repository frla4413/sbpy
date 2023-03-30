import pdb
import numpy as np
from sbpy.grid2d import MultiblockGrid, MultiblockSBP
from sbpy.utils import create_convergence_table, solution_to_file
from sbpy.meshes import get_annulus_grid, set_bd_info
from euler import solve_euler_ibvp, solve_euler_steady_state
import mms
from sbpy.grid2d import load_p3d

err_vec = []
acc     = 8
N       = 20
r_out   = 20
r_in    = 0.1
num_blocks = 15

blocks = get_annulus_grid(N, r_in, r_out, num_blocks)
grid   = MultiblockGrid(blocks)
sbp    = MultiblockSBP(grid, accuracy=acc)
t_end         = 5
dt            = 0.5e-1
num_timesteps = int(np.ceil(t_end/dt))


x_pos      = lambda r, th:  r*np.cos(th)
y_pos      = lambda r, th:  r*np.sin(th)
jump       = 2*np.pi/num_blocks

bd_info = {
        "bd1_x": [x_pos(r_out,jump*i) for i in range(2,14)],
        "bd1_y": [y_pos(r_out,jump*i) for i in range(2,14)],
        "bd2_x": [x_pos(r_out,jump*i) for i in range(-2,3)],
        "bd2_y": [y_pos(r_out,jump*i) for i in range(-2,3)],
        "bd3_x": [x_pos(r_in,jump*i) for i in range(15)],
        "bd3_y": [y_pos(r_in,jump*i) for i in range(15)],
        "bd4_x": {10},
        "bd4_y": {10}
        }


#bd_info = {
#        "bd1_x": [x_pos(r_out,jump*i) for i in range(2,9)],
#        "bd1_y": [y_pos(r_out,jump*i) for i in range(2,9)],
#        "bd2_x": [x_pos(r_out,jump*i) for i in range(-2,3)],
#        "bd2_y": [y_pos(r_out,jump*i) for i in range(-2,3)],
#        "bd3_x": [x_pos(r_in,jump*i) for i in range(10)],
#        "bd3_y": [y_pos(r_in,jump*i) for i in range(10)],
#        "bd4_x": {10},
#        "bd4_y": {10}
#        }


#grid.plot_domain(boundary_indices=True)
boundary_condition = {
        "bd1" : "inflow",
        "bd2" : "outflow",
        "bd3" : "wall",
        "bd4" : "None",
        }

set_bd_info(grid, bd_info, boundary_condition)
for (bd_idx, (block_idx, side)) in enumerate(grid.get_boundaries()):
    print("bd: " + str(bd_idx) + "," + str(grid.get_boundary_info(bd_idx)))

initu = []
initv = []
initp = []
for (X,Y) in blocks:
    initu.append(np.array(np.ones(X.shape)))
    initv.append(np.array(np.zeros(X.shape)))
    initp.append(np.array(np.zeros(X.shape)))

u_bar = 1
v_bar = 0
p_bar = 0

def wn_data(sbp, block_idx, side, t):
    n        = sbp.get_normals(block_idx,side)
    nx       = n[:,0] 
    ny       = n[:,1] 
    xbd, ybd = sbp.grid.get_boundary(block_idx,side)
    return u_bar*nx + v_bar*ny

def wt_data(sbp, block_idx, side, t):
    n        = sbp.get_normals(block_idx,side)
    nx       = n[:,0] 
    ny       = n[:,1] 
    xbd, ybd = sbp.grid.get_boundary(block_idx,side)
    return - u_bar*ny + v_bar*nx

def p_data(sbp, block_idx, side, t):
    xbd, ybd = sbp.grid.get_boundary(block_idx,side)
    return p_bar


bd_data = {
    "wn_data": wn_data,
    "wt_data": wt_data,
    "p_data" : p_data
    }

name = 'plots/sol'
U,V,P = solve_euler_ibvp(sbp, boundary_condition, bd_data, initu, initv, initp,\
                         dt, num_timesteps, name_base = name)

