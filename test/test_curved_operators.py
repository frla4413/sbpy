import unittest

import numpy as np
import sbpy.operators
from sbpy.utils import get_bump_grid
from sbpy.grid2d import MultiblockGrid, MultiblockSBP, get_center
import pdb


N_vec = [41,81,161]
err = []
err2 = []

for i in range(len(N_vec)):
#    X,Y = get_bump_grid(N_vec[i])
    N = N_vec[i]
#    X,Y = np.meshgrid(np.linspace(0,1,N), np.linspace(0,1,N))
#    X = np.transpose(X)
#    Y = np.transpose(Y)
#    grid = MultiblockGrid([(X,Y)])
#    sbp = MultiblockSBP(grid, accuracy=4)
#    sin_vec = np.sin(2*np.pi*Y)
#    cos_vec = np.cos(2*np.pi*Y)
#
#    err_vec = sbp.diffy([sin_vec])[0] - 2*np.pi*cos_vec
#    err.append(np.sqrt(sbp.integrate([err_vec*err_vec])))

    x = np.linspace(0, 1,N)
    dx = x[1] - x[0]
    sbp_op = sbpy.operators.SBP1D(N, dx, accuracy = 4)
    D = sbp_op.D
    P = sbp_op.P
    err_vec = np.abs(D*np.exp(-x) + np.exp(-x))
    err2.append(np.sqrt(err_vec@P@err_vec))


#rates = np.log2(np.array(err[:-1])/np.array(err[1:]))
rates = np.log2(np.array(err2[:-1])/np.array(err2[1:]))
print(rates)
