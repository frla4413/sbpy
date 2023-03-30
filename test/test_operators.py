import unittest

import numpy as np
import sbpy.operators
from sbpy import grid2d

class Test1DOperators(unittest.TestCase):

    def setUp(self):
        self.N = 100
        self.x = np.linspace(0, 1, self.N)
        self.dx = 1/(self.N-1)
        self.sbp_op = sbpy.operators.SBP1D(self.N, self.dx)
        self.D = self.sbp_op.D
        self.P = self.sbp_op.P
        self.Q = self.sbp_op.Q


    def test_differential_accuracy(self):
        tol = 1e-14
        self.assertTrue(np.max(np.abs(self.D@self.x - 1.0)) < tol)
        self.assertTrue(np.max(np.abs(self.D@np.ones(self.N))) < tol)


    def test_integral_accuracy(self):
        tol = 1e-14
        self.assertTrue(np.max(np.abs(sum(self.P@self.x) - 0.5)) < tol)
        self.assertTrue(np.max(np.abs(sum(self.P@np.ones(self.N)) - 1.0)) < tol)


    def test_sbp_property(self):
        tol = 1e-14
        E        = np.zeros((self.N,self.N))
        E[0,0]   = -1
        E[-1,-1] = 1
        self.assertTrue(np.all(self.Q + np.transpose(self.Q) == E))


class Test2DOperators(unittest.TestCase):

    def setUp(self):
        (X_blocks, Y_blocks) = grid2d.load_p3d('test/cyl50.p3d')
        self.X = X_blocks[0]
        self.Y = Y_blocks[0]
        self.sbp_op = sbpy.operators.SBP2D(self.X,self.Y)


    def test_differential_accuracy(self):
        tol = 1e-10
        #errx = np.max(np.abs(self.sbp_op.diffx(np.ones(self.X.shape))))
        #erry = np.max(np.abs(self.sbp_op.diffy(np.ones(self.Y.shape))))
        errx = np.max(np.abs(self.sbp_op.diffx(self.X - 1)))
        erry = np.max(np.abs(self.sbp_op.diffy(self.Y - 1)))
        err = np.fmax(errx,erry)
        print(err)
        print(self.sbp_op.diffy(self.Y))
        self.assertTrue(err < tol)


if __name__ == '__main__':
    unittest.main()
