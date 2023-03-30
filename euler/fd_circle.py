""" Example file for solving Euler on a circular grid with FD-operators"""

import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sbpy.meshes import get_annulus_dg_grid, get_annulus_grid
from sbpy.grid2d import MultiblockGrid, MultiblockDGSBP, MultiblockSBP

from sbpy.utils import create_convergence_table, solution_to_file, get_gauss_initial_data
from euler import euler_operator, wall_operator, outflow_operator, pressure_operator, inflow_operator, outflow_operator, force_operator, interface_operator, ivp_solution_to_euler_variables, solve_euler
from scipy import sparse
from sbpy.grid2d import load_p3d
from sbpy import dg_operators
from sbpy import grid2d
from solve import solve_ivp

dt            = 1e-3
num_timesteps = 10
acc           = 2

blocks = get_annulus_grid(30,4)
grid   = MultiblockGrid(blocks)
sbp    = MultiblockSBP(grid,accuracy=acc)

#grid.plot_domain(boundary_indices=True)
indices = {
        "inflow_idx"  : {2,4},
        "pressure_idx": {0,6},
        "outflow_idx" : {},
        "wall_idx"    : {1,3,5,7}
        }
initu = []
initv = []
initp = []
u_bar = 1
v_bar = 0
p_bar = 0

for (X,Y) in blocks:
    initu.append(u_bar*np.array(np.ones(X.shape)))
    initv.append(np.array(np.zeros(X.shape)))
    initp.append(np.array(np.zeros(X.shape)))

def get_wn_data(sbp, block_idx, side,t):
    n       = sbp.get_normals(block_idx,side)
    nx      = n[:,0] 
    ny      = n[:,1] 
    xbd,ybd   = sbp.grid.get_boundary(block_idx,side)
    return nx*u_bar + ny*v_bar

def get_wt_data(sbp, block_idx, side,t):
    n       = sbp.get_normals(block_idx,side)
    nx      = n[:,0] 
    ny      = n[:,1] 
    xbd,ybd = sbp.grid.get_boundary(block_idx,side)
    return -ny*u_bar + ny*v_bar

bd_data = {
        "wn_data": get_wn_data,
        "wt_data": get_wt_data
        }

U,V,P = solve_euler(sbp, indices, bd_data, initu, initv, initp, dt, num_timesteps)
solution_to_file(grid,U,V,P,'mms_test/mms_test')
