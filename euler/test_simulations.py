import unittest

import pdb
import scipy
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt

from sbpy.utils import get_circle_sector_grid, get_bump_grid, create_convergence_table, solution_to_tec
from sbpy.grid2d import MultiblockGrid, MultiblockSBP, get_center
from euler import euler_operator, wall_operator, solve, outflow_operator, interior_penalty_operator_w
from sbpy.artificial_dissipation_operators import generate_artificial_dissipation_matrix_2D


def get_gauss_initial_data(X, Y, cx, cy):
    gauss_bell = np.array([np.exp(-10*(X-cx)**2)])
    initu = gauss_bell*np.array([np.exp(-10*(Y-cy)**2)])
    initv = np.array([np.zeros(X.shape)])
    initw = np.array([np.zeros(X.shape)])
    initp = np.array([np.ones(X.shape)])
    return initu, initv, initw, initp

def square_cavity_flow_polar_coordinates(
        N = 30,
        num_timesteps = 10,
        dt = 0.1):

    (X,Y) = np.meshgrid(np.linspace(0,3,N), np.linspace(0,2*np.pi,N))
    X = np.transpose(X)
    Y = np.transpose(Y)
    X = X[:,:-1]
    Y = Y[:,:-1]
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid,periodic=True)
    initu, initv, initw, initp = get_gauss_initial_data(X, Y, 1, np.pi)
#    initu = initv
#    initv = [X*10*np.exp(-30*(1-X)**2)*np.sin(Y)]
    plt.quiver(X,Y,initu[0],initv[0])
    plt.show()
#    art_diss_matrix = generate_artificial_dissipation_matrix_2D(grid, alpha = 10)
#    art_diss_matrix = scipy.sparse.kron(scipy.sparse.eye(4), art_diss_matrix)

    def spatial_op(state):
        L,J = euler_operator(sbp, state) + \
            wall_operator(sbp, state, 0, 'e') +\
            interior_penalty_operator_w(sbp, state, 0)
#            outflow_operator(sbp, state, 0, 's') +\
#            outflow_operator(sbp, state, 0, 'n') +\
#        return L
#        L+= art_diss_matrix@state
#        J+= art_diss_matrix
        return np.array([L, J], dtype=object)

    filename = "sol/sol"
    U,V,W,P = solve(grid, spatial_op, initu,
                    initv, initw, initp, dt, num_timesteps, filename)

    return grid,U,V,W,P,dt

#-----------------------------------
#circle_sector_cavity_flow()

if __name__ == '__main__':
    N = 40
    grid,U,V,W,P,dt = square_cavity_flow_polar_coordinates(N=N,dt= 1e-2,
                                                           num_timesteps=200)
