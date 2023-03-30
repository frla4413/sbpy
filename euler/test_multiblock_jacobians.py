import unittest
import pdb

import numpy as np
from scipy import sparse
from scipy.optimize import approx_fprime

from sbpy.meshes import get_circle_sector_grid, get_bump_grid,get_cylinder_channel_grid
from sbpy.grid2d import MultiblockGrid, MultiblockSBP, collocate_corners
from euler import euler_operator, wall_operator, outflow_operator, pressure_operator, inflow_operator, interface_operator, jans_inflow_operator 

from sbpy.dg_operators import DGSBP1D,legendre_gauss_lobatto_nodes_and_weights, CurveInterpolant
from sbpy.meshes import transfinite_quad_map, get_cylinder_channel_grid
from grid2d import flatten_multiblock_vector, allocate_gridfunction
## to run all tests: python test_jacobians.py

## run one test:python -m test_multiblock_jacobians TestJacobians.test_pressure_jacobian


class TestJacobians(unittest.TestCase):

    N             = 3
    blocks = []
    x = np.linspace(-1,0,N)
    y = np.linspace(0,1,N)
    [X,Y]  = np.meshgrid(x,y)
    blocks.append([np.transpose(X),np.transpose(Y)])

    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)
    [X,Y]  = np.meshgrid(x,y)
    blocks.append([X,Y])
    collocate_corners(blocks)

    grid = MultiblockGrid(blocks)
    sbp  = MultiblockSBP(grid)

#    U = []
#    V = []
#    P = []
#
#    for (X,Y) in blocks:
#        U.append(np.exp(-X)*np.cos(Y*X))
#        V.append(np.sin(Y*X))
#        P.append(np.exp(-X)*np.exp(-Y*X))
#
#    U = flatten_multiblock_vector(U)
#    V = flatten_multiblock_vector(V)
#    P = flatten_multiblock_vector(P)

    U = flatten_multiblock_vector([(np.exp(-X)@np.cos(Y*X)).flatten() for (X,Y) in blocks])
    V = np.concatenate([ np.sin(Y*X).flatten() for (X,Y) in blocks])
    P = np.concatenate([ (np.exp(-X)@np.exp(-Y*X)).flatten() for (X,Y) in blocks])

    state = np.array([U, V, P]).flatten()

    def test_interface_jacobian(self):

        if_idx                           = 0
        [((idx1, side1), (idx2, side2))] = self.grid.get_interfaces()
        S, J                             = interface_operator(self.sbp, self.state, \
                                            idx1, side1, idx2, side2, if_idx)
        J                                = J.todense()
        J_approx                         = np.zeros(J.shape)

        for i,grad in enumerate(J):
            f = lambda x: interface_operator(self.sbp, x, idx1, side1, \
                                             idx2, side2, if_idx)[0][i]

            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact  = J[i,:]
            J_approx[i,:] = grad_approx
            err         = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Interface SAT, Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)


    def test_euler_jacobian(self):

        S,J  = euler_operator(self.sbp, self.state)

        for i,grad in enumerate(J):
            f = lambda x: euler_operator(self.sbp, x)[0][i]
            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact = J[i,:]
            err = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Euler OP, Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)


    def test_wall_jacobian(self):

        e         = 0
        side      = 'e'
        block_idx = 1
        S, J = wall_operator(self.sbp, self.state, block_idx, side)
        J = J.todense()

        for i,grad in enumerate(J):
            f = lambda x: wall_operator(self.sbp, x, block_idx, side)[0][i]

            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact = J[i,:]
            err = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Wall SAT, Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)
    
    def test_pressure_jacobian(self):

        e         = 0
        side      = 'w'
        block_idx = 1

        S, J = pressure_operator(self.sbp, self.state, block_idx, side)
        J                = J.todense()


        for i,grad in enumerate(J):
            def f(x):
                return pressure_operator(self.sbp, x, block_idx, side)[0][i]

            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact  = J[i,:]
            err         = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Pressure SAT, Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)


    def test_inflow_jacobian(self):

        e         = 0
        side      = 'e'
        block_idx = 1

        S, J = inflow_operator(self.sbp, self.state, block_idx, side, -1, 1,e)
        J = J.todense()

        for i,grad in enumerate(J):
            f           = lambda x: inflow_operator(self.sbp, x, block_idx, side, -1, 1,e)[0][i]
            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact  = J[i,:]
            err         = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Inflow SAT, Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)


    def test_jans_inflow_jacobian(self):

        side      = 'e'
        block_idx = 1
        S, J = jans_inflow_operator(self.sbp, self.state, block_idx, side, data = np.pi)
        J = J.todense()

        for i,grad in enumerate(J):
            f = lambda x: jans_inflow_operator(self.sbp, x, block_idx, 
                                               side, data=np.pi)[0][i]

            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact = J[i,:]
            err = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Wall SAT, Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)



    def test_outflow_jacobian(self):

        e         = 0
        side      = 'e'
        block_idx = 1

        S, J,ux,uy,vx,vy = euler_operator(self.sbp, self.state, e)
        S, J             = outflow_operator(self.sbp, self.state, block_idx, \
                                side, ux, uy, vx, vy, e)
        J                = J.todense()

        for i,grad in enumerate(J):
            def f(x):
                S, J,ux,uy,vx,vy = euler_operator(self.sbp, x, e)
                return outflow_operator(self.sbp, x, block_idx, side, \
                                        ux, uy, vx, vy, e)[0][i]
            grad_approx          = approx_fprime(self.state, f, 1e-8)
            grad_exact           = J[i,:]
            err                  = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Outflow SAT, Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)

if __name__ == '__main__':
    unittest.main()
