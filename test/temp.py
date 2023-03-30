import sys
sys.path.append('/Users/oskal44/dev/')
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sbpy import operators
from sbpy import grid2d
from sbpy import multiblock_solvers
from sbpy import animation
from sbpy import utils


N = 21
blocks = utils.get_annulus_grid(N)
grid2d.collocate_corners(blocks)
grid = grid2d.MultiblockGridSBP(blocks, accuracy=4)
init = [ np.zeros(shape) for shape in grid.get_shapes() ]
#for (k, (X,Y)) in enumerate(grid.get_blocks()):
#    init[k] = 0.1*norm.pdf(X,loc=-0.5,scale=0.2)*norm.pdf(Y,loc=0.5,scale=0.2)

def g(t,x,y):
    return 1

def h(t,x,y):
    return 0

velocity = np.array([1,1])/np.sqrt(2)
diffusion = 0.01

solver = multiblock_solvers.AdvectionDiffusionSolver(grid, initial_data=init,
                                                     velocity=velocity,
                                                     diffusion=diffusion)
solver.set_boundary_condition(1,{'type': 'dirichlet', 'data': g})
solver.set_boundary_condition(3,{'type': 'dirichlet', 'data': g})
solver.set_boundary_condition(5,{'type': 'dirichlet', 'data': g})
solver.set_boundary_condition(7,{'type': 'dirichlet', 'data': g})
tspan = (0.0, 4.0)
import time

start = time.time()
solver.solve(tspan)
end = time.time()
print("Elapsed time: " + str(end - start))

U = []
for frame in np.transpose(solver.sol.y):
    U.append(grid2d.array_to_multiblock(grid, frame))

animation.animate_multiblock(grid, U, stride=1)
