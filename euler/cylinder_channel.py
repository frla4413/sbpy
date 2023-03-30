import pdb
import numpy as np
from sbpy.grid2d import MultiblockGrid, MultiblockSBP, MultiblockDGSBP, collocate_corners
from sbpy.utils import create_convergence_table, solution_to_file
from sbpy.meshes import set_bd_info, get_cylinder_channel_grid
from euler import solve_euler_ibvp, solve_euler_steady_state
import mms
from sbpy.grid2d import load_p3d

acc           = 4
dt            = 1/10
num_timesteps = 40

blocks, bd_labels = get_cylinder_channel_grid(80, method = 'fd')
grid = MultiblockGrid(blocks)
#grid.plot_domain(boundary_indices=True)

#print(grid.get_interfaces())
for i in range(len(grid.get_interfaces())):
    if grid.is_flipped_interface(i):
        print('flipped interfaces: ' + str(i))

grid.plot_grid()
sbp    = MultiblockSBP(grid, accuracy=acc)

bd_info = {
        "bd1_x": bd_labels["top_x"],
        "bd1_y": bd_labels["top_y"],
        "bd2_x": bd_labels["bottom_x"],
        "bd2_y": bd_labels["bottom_y"],
        "bd3_x": bd_labels["cylinder_x"]+bd_labels["left_x"]+bd_labels["right_x"],
        "bd3_y": bd_labels["cylinder_y"]+bd_labels["left_y"]+bd_labels["right_y"],
        "bd4_x": [10],
        "bd4_y": [10]
        }

boundary_condition = {
        "bd1" : "inflow",
        "bd2" : "pressure",
        "bd3" : "wall",
        "bd4" : "outflow"
        }

set_bd_info(grid, bd_info, boundary_condition)

initu = []
initv = []
initp = []
for (X,Y) in blocks:
    initu.append(np.zeros(X.shape))
    initv.append(-np.ones(X.shape))
    initp.append(np.zeros(X.shape))

def wn_data(sbp, block_idx, side,t):
    n        = sbp.get_normals(block_idx,side)
    nx       = n[:,0] 
    ny       = n[:,1] 
    xbd, ybd = grid.get_boundary(block_idx,side)
    return -ny

def wt_data(sbp, block_idx, side,t):
    n        = sbp.get_normals(block_idx,side)
    nx       = n[:,0] 
    ny       = n[:,1] 
    xbd, ybd = grid.get_boundary(block_idx,side)
    return nx

bd_data = {
    "wn_data": wn_data,
    "wt_data": wt_data,
    "p_data" : 0
    }

name = 'plots/sol'
solve_euler_ibvp(sbp, boundary_condition, bd_data, initu, initv, initp, \
                         dt, num_timesteps, name_base = name)

#U,V,P = solve_euler_steady_state(sbp, boundary_condition, bd_data, initu, initv, initp)

#solution_to_file(grid,U,V,P,'cylinder_sol_steady/cyl')
