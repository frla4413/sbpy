import unittest
import pdb

import numpy as np
from scipy import sparse
from scipy.optimize import approx_fprime

from sbpy.utils import get_circle_sector_grid, get_bump_grid
from sbpy.grid2d import MultiblockGrid, MultiblockSBP

from adv_diff import spatial_op, sat_op, spatial_jacobian, sat_jacobian_south, sat_jacobian_north
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
    U = np.sin(X)*np.sin(Y)
    state = U.flatten()
    a = 1.2
    b = 1.3
    e = 0.8
    t = 0.86

    def test_adv_diff_jacobian(self):
        t = self.t
        a = self.a
        b = self.b
        e = self.e
        
        J = spatial_jacobian(self.sbp, a, b, e)

        for i,grad in enumerate(J):
            f = lambda x: spatial_op(t,x,self.sbp,a,b,e)[0][i]
            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact = J[i,:]
            err = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("Adv-diff OP, Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)

    def test_sat_jacobian_south(self):
        t = self.t
        a = self.a
        b = self.b
        e = self.e
        
        J = sat_jacobian_south(self.sbp, a, b, e)
        J = J.todense()

        for i,grad in enumerate(J):
            def f(x):
                uy = spatial_op(t,x,self.sbp,a,b,e)[1]
                return sat_op(t,x,uy,self.sbp,b,e,'s')[i]

            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact = J[i,:]
            err = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("SOUTH SAT OP, Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)


    def test_sat_jacobian_north(self):
        t = self.t
        a = self.a
        b = self.b
        e = self.e
        
        J = sat_jacobian_north(self.sbp, a, b, e)
        J = J.todense()

        for i,grad in enumerate(J):
            def f(x):
                uy = spatial_op(t,x,self.sbp,a,b,e)[1]
                return sat_op(t,x,uy,self.sbp,b,e,'n')[i]

            grad_approx = approx_fprime(self.state, f, 1e-8)
            grad_exact = J[i,:]
            err = np.linalg.norm(grad_approx-grad_exact, ord=np.inf)
            print("NORTH SAT OP, Gradient error = {:.2e}".format(err))
            self.assertTrue(err < 1e-5)


if __name__ == '__main__':
    unittest.main()
