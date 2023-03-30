import unittest
import pdb

import numpy as np
from scipy import sparse
from scipy.optimize import approx_fprime

from sbpy.utils import get_circle_sector_grid
from sbpy.grid2d import MultiblockGrid, MultiblockSBP
from euler import euler_operator, wall_operator, outflow_operator, pressure_operator, inflow_operator, outflow_operator, pressure_inflow_operator

class TestJacobians(unittest.TestCase):

    (X,Y) = get_circle_sector_grid(3, 0.0, 3.14/2, 0.2, 1.0)
    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid)
    U = X
    V = Y
    P = X**2 + Y**2
    state = np.array([U, V, P]).flatten()

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

        data = lambda t: 0
        S, J = wall_operator(self.sbp, self.state, 0, 'w')
        J = J.todense()


        for i,grad in enumerate(J):
            f = lambda x: wall_operator(self.sbp, x, 0, 'w')[0][i]
            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact = J[i,:]
            err = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Wall SAT, Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)


    def test_pressure_jacobian(self):
        S, J = pressure_operator(self.sbp, self.state, 0, 'w')
        J = J.todense()

        for i,grad in enumerate(J):
            f = lambda x: pressure_operator(self.sbp, x, 0, 'w')[0][i]
            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact = J[i,:]
            err = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Pressure SAT, Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)


    def test_inflow_jacobian(self):
        S, J = inflow_operator(self.sbp, self.state, 0, 'w', -1, 1)
        J = J.todense()

        for i,grad in enumerate(J):
            f = lambda x: inflow_operator(self.sbp, x, 0, 'w', -1, 1)[0][i]
            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact = J[i,:]
            err = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Inflow SAT, Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)


    def test_outflow_jacobian(self):
        S, J = outflow_operator(self.sbp, self.state, 0, 'w',0)
        J = J.todense()

        for i,grad in enumerate(J):
            f = lambda x: outflow_operator(self.sbp, x, 0, 'w',0)[0][i]
            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact = J[i,:]
            err = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Outflow SAT, Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)

    def test_pressure_inflow_jacobian(self):
        S, J = pressure_inflow_operator(self.sbp, self.state, 0, 'w',0.5,-1)
        J = J.todense()

        for i,grad in enumerate(J):
            f = lambda x: pressure_inflow_operator(self.sbp, x, 0, 'w', 0.5,-1)[0][i]
            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact = J[i,:]
            err = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Inflow SAT, Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)

if __name__ == '__main__':
    unittest.main()

#to run one test
#python test_jacobians.py TestJacobians.test_outflow_jacobian
#to run all tests
#python test_jacobians.py
