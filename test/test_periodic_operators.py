import unittest
import pdb

import numpy as np
import sbpy.operators
from sbpy import grid2d

class Test1DOperators(unittest.TestCase):

    def setUp(self):
        self.N = 100
        self.x = np.linspace(0, 1, self.N)
        self.x = self.x[:-1]
        self.dx = 1/(self.N-1)
        self.N = len(self.x)
        self.sbp_op = sbpy.operators.SBP1D(self.N, self.dx, periodic = True)
        self.D = self.sbp_op.D
        self.P = self.sbp_op.P
        self.Q = self.sbp_op.Q

    def test_differential_accuracy(self):
        ''' Periodic operators only accurate in the interior '''
        tol = 1e-14
        x = self.x
        D_x = self.D @ x
        D_1 = self.D @ np.ones(self.N)
        self.assertTrue(np.max(np.abs(D_x[1:-1] - 1.0)) < tol)
        self.assertTrue(np.max(np.abs(D_1)) < tol)

    def test_sbp_property(self):
        E        = np.zeros((self.N,self.N))
        self.assertTrue(np.all(self.Q + np.transpose(self.Q) == E))

    def test_periodic_accuracy(self):
        ''' tol_diff depends on N '''

        tol_diff = 1e-2
        f = lambda x: np.sin(2*np.pi*x)
        f_x = lambda x: 2*np.pi*np.cos(2*np.pi*x)
        err_diff = np.linalg.norm(self.D @ f(self.x) - f_x(self.x), ord=np.inf)

        self.assertTrue(err_diff < tol_diff)

    def test_periodic_integration(self):
        ''' \int_0^1 sin(2 pi x) dx = 0 '''
        
        tol_int = 1e-14
        f = lambda x: np.sin(2*np.pi*x)
        err_const_int = np.ones(self.N)@ self.P @ np.ones(self.N) - 1
        err_intf = np.ones(self.N) @ self.P @ f(self.x)
        
        self.assertTrue(err_intf < tol_int)
        self.assertTrue(err_const_int < tol_int)


class Test2DOperators(unittest.TestCase):
    ''' 2D periodic operators. Exclude the last column in X and Y.
        Periodicity in the x-direction. 
    '''

    def setUp(self):
        self.N = 81
        (X,Y) = np.meshgrid(np.linspace(0,1,self.N), np.linspace(0,1,self.N))
        X = np.transpose(X)
        Y = np.transpose(Y)

        self.X = X[:-1]
        self.Y = Y[:-1]
        self.sbp_op = sbpy.operators.SBP2DPeriodic(self.X, self.Y)

    def test_periodic_accuracy(self):
        ''' exact for constant and 1-st order polynomial in the interior '''
        tol = 1e-12
        dx = self.sbp_op.diffx(self.X)
        errx = np.linalg.norm(dx[1:-2] - 1, ord=np.inf)
        erry = np.linalg.norm(self.sbp_op.diffy(self.Y) - 1, ord=np.inf)
        err = np.fmax(errx,erry)
        self.assertTrue(err < tol)

    def test_periodic_function(self):
        ''' test on periodic functions '''
        tol = 1e-2
        f = lambda x: np.sin(2*np.pi*x)
        f_p = lambda x: 2*np.pi*np.cos(2*np.pi*x)
        err_f = self.sbp_op.diffx(f(self.X)) - f_p(self.X)
        err_diffx = np.max(np.abs(err_f))
        err_f = self.sbp_op.diffy(f(self.Y)) - f_p(self.Y)
        err_diffy = np.max(np.abs(err_f))
        err = np.max([err_diffx,err_diffy])
        print(err_diffx)
        #self.assertTrue(err < tol)

    def test_sbp_property(self):
        Qx = self.sbp_op.P@self.sbp_op.Dx
        E = np.zeros(Qx.shape)
        QQTx= Qx + np.transpose(Qx)
        self.assertTrue(np.all(QQTx == E))

    def test_integration(self):
        ''' integral of f(x)  is 0 '''
        tol = 1e-12
        f   = lambda x: np.sin(2*np.pi*x)
        f_int = self.sbp_op.integrate(f(self.X))
        self.assertTrue(f_int < tol)

if __name__ == '__main__':
    unittest.main()
