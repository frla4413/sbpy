import sys
sys.path.append('../..')
import pickle

from sbpy import grid2d
from sbpy import multiblock_solvers
from sbpy import animation
from sbpy import utils
from mayavi import mlab

with open('highres_data/highres_sol161_29.pkl', 'rb') as f:
    U_highres,diffusion = pickle.load(f)

def g(t,x,y):
    return 1

velocity = np.array([1,1])/np.sqrt(2)
#diffusion = 0.01
tspan = (0.0, 3.5)

N = 21
blocks = utils.get_annulus_grid(N)
coarse_grid = grid2d.MultiblockGridSBP(blocks, accuracy=4)

N = 161
blocks = utils.get_annulus_grid(N)
fine_grid = grid2d.MultiblockGridSBP(blocks, accuracy=4)

nodes = utils.boundary_layer_selection(coarse_grid, [1,3,5,7], 4)

int_data = utils.fetch_highres_data(coarse_grid,
        nodes, fine_grid, U_highres, stride=8)

solver = multiblock_solvers.AdvectionDiffusionSolver(coarse_grid,
        velocity=velocity, diffusion=diffusion,
        internal_data = int_data, internal_indices = nodes)

solver.set_boundary_condition(1,{'type': 'dirichlet', 'data': g})
solver.set_boundary_condition(3,{'type': 'dirichlet', 'data': g})
solver.set_boundary_condition(5,{'type': 'dirichlet', 'data': g})
solver.set_boundary_condition(7,{'type': 'dirichlet', 'data': g})
solver.solve(tspan)

U = []
for frame in np.transpose(solver.sol.y):
    U.append(grid2d.array_to_multiblock(coarse_grid, frame))

animation.animate_multiblock(coarse_grid, U)
