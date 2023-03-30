"""This module contains functions for getting 
    artificial_dissipation to SBP operators."""

import pdb
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from sbpy import grid2d

def generate_artificial_dissipation_matrix_1D(N, dx, alpha = 1):
    
    stencil = np.array([-1, 1])
    MID = sparse.diags(stencil,
                       [0, 1],
                       shape=(N-1, N))

    data = np.array([-1, 1])
    row = np.array([0,0])
    col = np.array([0,1])
    TOP = sparse.bsr_matrix((data,(row,col)), shape =(1,N))

    D_tilde = sparse.vstack([TOP, MID])
    B     = np.ones(N)
    B[0]  = 0
    B     = sparse.diags(B)
    p     = np.ones(N)
    p[0]  = 0.5
    p[-1] = 0.5
    P_inv = 1/p
    P_inv_tilde = sparse.diags(P_inv)

    artificial_dissipation_matrix = alpha*dx*P_inv_tilde*np.transpose(D_tilde)*B*D_tilde
    return artificial_dissipation_matrix

def generate_artificial_dissipation_matrix_2D(grid, alpha = 1):

    shapes = grid.get_shapes()
    (Nx, Ny) = shapes[0]
    (X,Y) = grid.get_block(0)
    y = Y[0]
    dy = y[1] - y[0]
    x = X[:,0]
    dx = x[1] - x[0]
    art_diss_1d_y = generate_artificial_dissipation_matrix_1D(Ny, dy, alpha = alpha)
    art_diss_1d_x = generate_artificial_dissipation_matrix_1D(Nx, dx, alpha = alpha)
    return sparse.kron(art_diss_1d_x, art_diss_1d_y)
