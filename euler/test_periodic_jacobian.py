import unittest
import pdb

import numpy as np
from scipy import sparse
from scipy.optimize import approx_fprime

from sbpy.utils import get_circle_sector_grid, get_bump_grid
from sbpy.grid2d import MultiblockGrid, MultiblockSBP
from euler import euler_operator, wall_operator, outflow_operator, pressure_operator, inflow_operator, interior_low_operator, periodic_in_x_operator

## to run all tests: python test_jacobians.py
## to run one test : python -m test_jacobians TestJacobians.test_pressure_jacobian


class TestJacobians(unittest.TestCase):

    (X,Y) = np.meshgrid(np.linspace(0,2*np.pi,3), np.linspace(0,2*np.pi,3))
    X = np.transpose(X)
    Y = np.transpose(Y)
    X = X[:-1]
    Y = Y[:-1]

    grid = MultiblockGrid([(X,Y)])
    sbp = MultiblockSBP(grid, accuracy = 2, periodic = True)
    #grid.plot_domain()
    U = 1 + 0.1*np.sin(X)*np.sin(Y)
    V = 1 + np.cos(X)*np.cos(Y)
    P = U**2 + V**2
    state = np.array([U, V, P]).flatten()

    def test_euler_jacobian(self):
        e = 0.5
        
        S, J,ux,uy,vx,vy = euler_operator(self.sbp, self.state, e)

        for i,grad in enumerate(J):
            f = lambda x: euler_operator(self.sbp, x, e)[0][i]
            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact = J[i,:]
            err = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Euler OP, Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)

    def test_wall_jacobian(self):

        side = 'n'
        e = 0.5
        S, J = wall_operator(self.sbp, self.state, 0, side, e)
        J = J.todense()

        for i,grad in enumerate(J):
            f = lambda x: wall_operator(self.sbp, x, 0, side , e)[0][i]

            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact = J[i,:]
            err = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Wall SAT, Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)

    def test_pressure_jacobian(self):

        e = 0.6

        side = 'n'
        S, J,ux,uy,vx,vy = euler_operator(self.sbp, self.state)
        S, J = pressure_operator(self.sbp, self.state, 0, side,ux,uy,vx,vy, e)
        J = J.todense()

        for i,grad in enumerate(J):
            def f(x):
                S, J,ux,uy,vx,vy = euler_operator(self.sbp, x)
                return pressure_operator(self.sbp, x, 0, side,ux,uy,vx,vy,e)[0][i]

            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact = J[i,:]
            err = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Pressure SAT, Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)


    def test_inflow_jacobian(self):

        e = 1e-2
        
        S, J = inflow_operator(self.sbp, self.state, 0, 'w', -1, 1,e)
        J = J.todense()

        for i,grad in enumerate(J):
            f = lambda x: inflow_operator(self.sbp, x, 0, 'w', -1, 1,e)[0][i]
            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact = J[i,:]
            err = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Inflow SAT, Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)


    def test_outflow_jacobian(self):

        e = 1e-2

        S, J,ux,uy,vx,vy = euler_operator(self.sbp, self.state, e)
        S, J = outflow_operator(self.sbp, self.state, 0, 'w', ux, uy, vx, vy, e)
        J = J.todense()

        for i,grad in enumerate(J):
            def f(x):
                S, J,ux,uy,vx,vy = euler_operator(self.sbp, x, e)
                return outflow_operator(self.sbp, x, 0, 'w', ux, uy, vx, vy, e)[0][i]
            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact = J[i,:]
            err = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Outflow SAT, Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)

    def test_interior_jacobian(self):

        e = 1e-2
        lw_slice1 = slice(0,3,None)
        lw_slice2 = slice(0,3,1)

        S, J,ux,uy,vx,vy = euler_operator(self.sbp, self.state, e)
        S, J = interior_low_operator(self.sbp, self.state, 0, \
                                         lw_slice1, lw_slice2, uy, e)
        J = J.todense()


        for i,grad in enumerate(J):
            def f(x):
                S, J,ux,uy,vx,vy = euler_operator(self.sbp, x, e)
                return interior_low_operator(self.sbp, x, 0, \
                                         lw_slice1, lw_slice2, uy, e)[0][i]

            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact = J[i,:]
            err = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Outflow SAT, Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)

    def test_periodic_x_jacobian(self):
        turbulence = False

        e = 0

        S, J,ux,uy,vx,vy = euler_operator(self.sbp, self.state, e)
        S, J = periodic_in_x_operator(self.sbp, self.state, 0, ux, uy, vx, vy, \
                                         e, turb_model = turbulence)

        J = J.todense()

        for i,grad in enumerate(J):
            def f(x):
                S, J,ux,uy,vx,vy = euler_operator(self.sbp, x, e)
                return periodic_in_x_operator(self.sbp, x, 0, ux, uy, vx, vy, \
                                         e, turb_model = turbulence)[0][i]
            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact = J[i,:]
            err = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Outflow SAT, Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)

if __name__ == '__main__':
    unittest.main()
