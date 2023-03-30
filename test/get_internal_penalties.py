import sys
sys.path.append('..')
import numpy as np
from sbpy import grid2d
from sbpy import multiblock_solvers
from sbpy import utils


N = 161
blocks = utils.get_annulus_grid(N)
fine_grid = grid2d.MultiblockSBP(blocks, accuracy=4)

def g(t,x,y):
    return 1

velocity = np.array([1,1])/np.sqrt(2)
diffusion = 0.01

for (i,diffusion) in enumerate(np.linspace(0.001,1,30)):
    solver = multiblock_solvers.AdvectionDiffusionSolver(fine_grid,
                                                         velocity=velocity,
                                                         diffusion=diffusion)
    solver.set_boundary_condition(1,{'type': 'dirichlet', 'data': g})
    solver.set_boundary_condition(3,{'type': 'dirichlet', 'data': g})
    solver.set_boundary_condition(5,{'type': 'dirichlet', 'data': g})
    solver.set_boundary_condition(7,{'type': 'dirichlet', 'data': g})
    tspan = (0.0, 3.5)

    solver.solve(tspan)

    U = []
    for frame in np.transpose(solver.sol.y):
        U.append(grid2d.array_to_multiblock(fine_grid, frame))

    U_highres = U[-1]

    import pickle
    with open('highres_data/highres_sol161_'+str(i)+'.pkl', 'wb') as f:
        pickle.dump([U_highres, diffusion], f)

