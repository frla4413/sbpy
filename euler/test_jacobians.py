import unittest
import pdb

import numpy as np
from scipy import sparse
from scipy.optimize import approx_fprime

from sbpy.utils import get_circle_sector_grid
from sbpy.grid2d import MultiblockGrid, MultiblockSBP
from euler import euler_operator, wall_operator, outflow_operator

# to run:
#python -m test_jacobians TestJacobians.test_outflow_jacobian

class TestJacobians(unittest.TestCase):

    N = 3
    (X,Y) = np.meshgrid(np.linspace(0.4,1.5,N), np.linspace(0,1,N))
    X = np.transpose(X)
    Y = np.transpose(Y)

    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid)
    U = np.exp(-X)
    V = np.exp(-Y)
    W = np.exp(-X*Y)
    P = X**2 + Y**2
    state = np.array([U, V, W, P]).flatten()


    def test_euler_jacobian(self):
        S, J = euler_operator(self.sbp, self.state)

        for i,grad in enumerate(J):
            f = lambda x: euler_operator(self.sbp, x)[0][i]
            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact = J[i,:]
            err = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Euler OP, Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)


    def test_wall_jacobian(self):
        S, J = wall_operator(self.sbp, self.state, 0, 'e')
        J = J.todense()

        for i,grad in enumerate(J):
            f = lambda x: wall_operator(self.sbp, x, 0, 'e')[0][i]
            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact = J[i,:]
            err = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Wall SAT, Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)

    def test_outflow_jacobian(self):
        S, J = outflow_operator(self.sbp, self.state, 0, 's')
        J = J.todense()

        for i,grad in enumerate(J):
            f = lambda x: outflow_operator(self.sbp, x, 0, 's')[0][i]
            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact = J[i,:]
            err = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Outflow SAT, Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)


if __name__ == '__main__':
    unittest.main()

