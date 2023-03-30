import pdb
import numpy as np
from sbpy.grid2d import MultiblockGrid, MultiblockDGSBP, collocate_corners
from sbpy.utils import create_convergence_table, solution_to_file
from sbpy.meshes import set_bd_info, get_cylinder_grid
from euler import solve_euler_ibvp, solve_euler_steady_state
import mms
from sbpy.grid2d import load_p3d

order      = 6
r_out      = 51.3
r_in       = 0.1
num_blocks = 8

blocks        = get_cylinder_grid(order, num_blocks)
grid          = MultiblockGrid(blocks)
sbp           = MultiblockDGSBP(grid)
t_end         = 3
dt            = 1e-2
num_timesteps = 150#int(np.ceil(t_end/dt))

x_pos      = lambda r, th:  r*np.cos(th)
y_pos      = lambda r, th:  r*np.sin(th)
jump       = 2*np.pi/num_blocks

bd_info = {
        "bd1_x": [x_pos(r_out,jump*i) for i in range(3,9)],
        "bd1_y": [y_pos(r_out,jump*i) for i in range(3,9)],
        "bd2_x": [x_pos(r_out,jump*i) for i in range(0,4)],
        "bd2_y": [y_pos(r_out,jump*i) for i in range(0,4)],
        "bd3_x": [x_pos(r_in,jump*i) for i in range(9)],
        "bd3_y": [y_pos(r_in,jump*i) for i in range(9)],
        "bd4_x": {10},
        "bd4_y": {10}
        }

boundary_condition = {
        "bd1" : "inflow",
        "bd2" : "pressure",
        "bd3" : "wall",
        "bd4" : "None",
        }

set_bd_info(grid, bd_info, boundary_condition)
for (bd_idx, (block_idx, side)) in enumerate(grid.get_boundaries()):
    print("bd: " + str(bd_idx) + "," + str(grid.get_boundary_info(bd_idx)))
grid.plot_domain(boundary_indices=True)

angle = np.pi/4
u_bar = np.cos(angle)
v_bar = np.sin(angle)
p_bar = 0

initu = []
initv = []
initp = []
for (X,Y) in blocks:
    initu.append(u_bar*np.array(np.ones(X.shape)))
    initv.append(v_bar*np.array(np.ones(X.shape)))
    initp.append(np.array(np.zeros(X.shape)))


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
