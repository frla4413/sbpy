import pdb

import numpy as np
import matplotlib.pyplot as plt
from sbpy.meshes import get_bump_grid, get_circle_sector_grid, get_annulus_grid
from sbpy.grid2d import MultiblockGrid, MultiblockSBP
from sbpy.utils import create_convergence_table, solution_to_file
from euler import solve
from euler import euler_operator, wall_operator, outflow_operator, pressure_operator, inflow_operator, outflow_operator, solve, force_operator, interface_operator
from sbpy.euler.animation import animate_pressure, animate_velocity, animate_solution
import mms
from scipy import sparse
from sbpy.grid2d import load_p3d

N               = 40
dt              = 1e-2
num_timesteps   = 10
acc             = 4

blocks          = get_annulus_grid(N,7)

grid            = MultiblockGrid(blocks)
sbp             = MultiblockSBP(grid, accuracy=acc)

grid.plot_domain()
inflow_idx    = {8,6,4}
pressure_idx  = {}#{0,6}
outflow_idx   = {2,10,12,0}
wall_idx      = {1,3,5,7,9,11,13}

initu = []
initv = []
initp = []
u_bar = 1
v_bar = 0
p_bar = 0

for (X,Y) in blocks:
    initu.append(u_bar*np.array(np.ones(X.shape)))
    initv.append(v_bar*np.array(np.zeros(X.shape)))
    initp.append(p_bar*np.array(np.zeros(X.shape)))

def get_wn_data(sbp, block_idx, side,t):
    n       = sbp.get_normals(block_idx,side)
    nx      = n[:,0] 
    ny      = n[:,1] 
    xbd,ybd   = grid.get_boundary(block_idx,side)
    return nx*u_bar + ny*v_bar

def get_wt_data(sbp, block_idx, side,t):
    n       = sbp.get_normals(block_idx,side)
    nx      = n[:,0] 
    ny      = n[:,1] 
    xbd,ybd   = grid.get_boundary(block_idx,side)
    return -ny*u_bar + ny*v_bar 

def boundary_op(t,state,sbp):
    N = len(state)
    Sbd = np.zeros(N)
    Jbd = sparse.csr_matrix((N, N))

    for (bd_idx, (block_idx, side)) in enumerate(sbp.grid.get_boundaries()):
        (X,Y) = blocks[block_idx]

        if bd_idx in inflow_idx:
            S_bd, J_bd = inflow_operator(sbp, state, block_idx, side, \
                                    get_wn_data, get_wt_data,t)
        elif bd_idx in pressure_idx:
            S_bd, J_bd =   pressure_operator(sbp, state, block_idx, side)

        elif bd_idx in outflow_idx:
            S_bd, J_bd =   outflow_operator(sbp, state, block_idx, side)

        elif bd_idx in wall_idx:
            S_bd, J_bd = wall_operator(sbp, state, block_idx, side)

        Sbd += S_bd
        Jbd += J_bd
    return Sbd, Jbd

def interface_op(state):
    N = len(state)
    Sif = np.zeros(N)
    Jif = sparse.csr_matrix((N, N))

    for (idx1, side1), (idx2, side2) in grid.get_interfaces():
        S_if, J_if = interface_operator(sbp, state, idx1, side1, \
                                        idx2, side2)
        Sif += S_if
        Jif += J_if

    return Sif, Jif

def spatial_op(t,state):
    S,J      = euler_operator(sbp, state)
    Sbd, Jbd = boundary_op(t,state,sbp)
    Sif, Jif = interface_op(state)
    force    = force_operator(sbp,mms.force1,mms.force2,mms.force3,t)[0]

    return S + Sbd + Sif , J + Jbd + Jif

U,V,P = solve(grid, spatial_op, initu, initv, initp, dt, num_timesteps)
solution_to_file(grid,U,V,P,'mms_test/mms_test')
