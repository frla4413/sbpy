import sys
sys.path.append('../..')
import numpy as np
from sbpy import operators
from sbpy import grid2d
from sbpy import multiblock_solvers
from sbpy import utils


def u(t,x,y):
    return np.sin(t+x+y)


def ut(t,x,y):
    return np.cos(t+x+y)


def ux(t,x,y):
    return np.cos(t+x+y)


def uxx(t,x,y):
    return -np.sin(t+x+y)


def uy(t,x,y):
    return np.cos(t+x+y)


def uyy(t,x,y):
    return -np.sin(t+x+y)

errs = []
resolutions = np.array([11, 21, 41, 81])
h = 1/(resolutions-1)

for N in resolutions:
    blocks = [utils.get_circle_sector_grid(N, 0, 0.5*np.pi, 0.2, 1.0),
              utils.get_circle_sector_grid(N, 0.5*np.pi, np.pi, 0.2, 1.0),
              utils.get_circle_sector_grid(N, np.pi, 1.5*np.pi, 0.2, 1.0),
              utils.get_circle_sector_grid(N, 1.5*np.pi, 2*np.pi, 0.2, 1.0)]
    grid2d.collocate_corners(blocks)
    grid = grid2d.MultiblockGridSBP(blocks, accuracy=4)
    solver = multiblock_solvers.AdvectionDiffusionSolver(grid,
                                                         u=u, ux=ux, uxx=uxx,
                                                         uy=uy, uyy=uyy, ut=ut)
    tspan = (0.0, 1.05)
    err = solver.run_mms_test(tspan)
    print("\n" + str(err)+"\n")
    errs.append(err)

utils.create_convergence_table(resolutions, errs, h)

