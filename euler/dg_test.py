""" Example file for runnging a FD circle """

import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sbpy.meshes import get_annulus_dg_grid, get_annulus_grid
from sbpy.grid2d import MultiblockGrid, MultiblockDGSBP, MultiblockSBP

from scipy import sparse
from sbpy import dg_operators

nodes,weights = dg_operators.legendre_gauss_lobatto_nodes_and_weights(4)
t             = 0.5
dt            = 0.05
nodes         = 0.5*dt*nodes + 0.5*dt + t
weights       = weights*0.5*dt
D             = dg_operators.polynomial_derivative_matrix(nodes)
P             = sparse.diags([weights], [0])
Q             = P@D
