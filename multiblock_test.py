import unittest

from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt

from sbpy.utils import get_circle_sector_grid, get_bump_grid, get_annulus_grid
from sbpy.grid2d import MultiblockGrid, MultiblockSBP, get_center

N = 11

(X,Y) = np.meshgrid(np.linspace(0,1,N), np.linspace(0,1,N))
X = np.transpose(X)
Y = np.transpose(Y)
grid = MultiblockGrid([(X,Y), (X+1,Y), (X+2,Y)])

grid.plot_grid()
