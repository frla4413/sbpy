"""This module contains functions for getting SBP operators."""

import pdb

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from sbpy import grid2d

class SBP1D:
    """ Class representing a 1D finite difference SBP operator.

    Attributes:
        P: Quadrature matrix.
        Q: An almost skew-symmetric matrix (Q+Q^T = diag(-1,0,0,...,1)), such
            that P^(-1)Q is an SBP operator.
        D: The SBP operator P^(-1)Q.
    """

    def __init__(self, N, dx, accuracy = 2):
        """ Initializes an SBP1D object.

        Args:
            N: The number of grid points.
            dx: The spacing between the grid points.
            accuracy: The accuracy of the interior stencil (2 or 4)
        """

        assert(accuracy in [2,4,8])

        self.N  = N
        self.dx = dx

        if accuracy == 2:
            stencil = np.array([-0.5, 0.0, 0.5])
            MID = sparse.diags(stencil,
                               [0, 1, 2],
                               shape=(N-2, N))

            TOP = sparse.bsr_matrix(([-0.5, 0.5], ([0, 0], [0, 1])),
                                    shape=(1, N))
            BOT = sparse.bsr_matrix(([-0.5, 0.5], ([0, 0], [N-2, N-1])),
                                    shape=(1, N))

            self.Q = sparse.vstack([TOP, MID, BOT])

            p     = np.ones(self.N)
            p     = dx*p
            p[0]  = 0.5*dx
            p[-1] = 0.5*dx
            p_inv = 1/p

        if accuracy == 4:
            h1=17/48
            h2=59/48
            h3=43/48
            h4=49/48
            q1=1/12
            q2=2/3
            Q = sparse.diags([(N-2)*[q1], (N-1)*[-q2], N*[0], (N-1)*[q2], (N-2)*[-q1]],
                             [-2,-1,0,1,2])
            Q = Q.tolil()
            Q[:4,:4] = [[-1/2  , 59/96 , -1/12 , -1/32],
                        [-59/96, 0     , 59/96 , 0    ],
                        [1/12  , -59/96, 0     , 59/96],
                        [1/32  , 0     , -59/96, 0    ]]
            Q[-4:,-4:] = -np.rot90(Q[:4,:4].todense(), k=2)
            self.Q = Q
            p = dx*np.concatenate([[h1,h2,h3,h4], np.ones(N-8), [h4, h3, h2, h1]])
            p_inv = 1/p
            
        if accuracy == 8:
            # from sbp_opmod.m
            Nx = N
            p_block = np.array([1498139/5080320, 1107307/725760, 
                               20761/80640, 1304999/725760, 299527/725760, 103097/80640,
                               670091/725760, 5127739/5080320])
            
            p     = dx*np.concatenate([p_block,np.ones(Nx-16),p_block[::-1]])
            p_inv = 1/p
            
            r67 = 0.649
            r68 = -0.104
            r78 = 0.755
            
            block1= np.array([-2540160/1498139, 
                              -142642467/5992556 + 50803200/1498139*r78+ 
                               5080320/1498139*r67 + 25401600/1498139*r68,
                               705710031/5992556-228614400/1498139*r78 - 
                               25401600/1498139*r67-121927680/1498139*r68,
                              -3577778591/17977668+381024000/1498139*r78+
                               50803200/1498139*r67+228614400/1498139*r68,
                               203718909/1498139-254016000/1498139*r78
                              -50803200/1498139*r67-203212800/1498139*r68,
                              -32111205/5992556+25401600/1498139*r67+
                               76204800/1498139*r68,
                              -652789417/17977668+76204800/1498139*r78-
                               5080320/1498139*r67,
                               74517981/5992556-25401600/1498139*r78- 
                               5080320/1498139*r68, 0, 0, 0, 0])

            block2 = np.array([142642467/31004596- 
                               7257600/1107307*r78-725760/1107307*r67- 
                               3628800/1107307*r68, 0, 
                              -141502371/2214614+91445760/1107307*r78+
                               10886400/1107307*r67+50803200/1107307*r68,
                               159673719/1107307-203212800/1107307*r78-
                               29030400/1107307*r67-127008000/1107307*r68, 
                              -1477714693/13287684+152409600/1107307*r78+
                               32659200/1107307*r67+127008000/1107307*r68,
                               11652351/2214614-17418240/1107307*r67-
                               50803200/1107307*r68, 
                               36069450/1107307-50803200/1107307*r78+
                               3628800/1107307*r67,
                              -536324953/46506894+17418240/1107307*r78+
                               3628800/1107307*r68, 0, 0, 0, 0])


            block3 = np.array([-18095129/134148+3628800/20761*r78+
                                403200/20761*r67+1935360/20761*r68, 
                                47167457/124566-10160640/20761*r78-
                                1209600/20761*r67-5644800/20761*r68, 0, 
                               -120219461/124566+25401600/20761*r78+
                                4032000/20761*r67+16934400/20761*r68, 
                                249289259/249132-25401600/20761*r78-
                                6048000/20761*r67-22579200/20761*r68,
                               -2611503/41522+3628800/20761*r67+10160640/20761*r68, 
                               -7149666/20761+10160640/20761*r78-806400/20761*r67,
                                37199165/290654-3628800/20761*r78-806400/20761*r68, 
                                0, 0, 0, 0])
            

            block4 = np.array([3577778591/109619916- 54432000/1304999*r78-
                               7257600/1304999*r67-32659200/1304999*r68, 
                              -159673719/1304999+203212800/1304999*r78+
                               29030400/1304999*r67+127008000/1304999*r68,
                               360658383/2609998-228614400/1304999*r78-
                               36288000/1304999*r67-152409600/1304999*r68, 0,
                              -424854441/5219996+127008000/1304999*r78+
                               36288000/1304999*r67+127008000/1304999*r68, 
                               22885113/2609998-29030400/1304999*r67-
                               76204800/1304999*r68, 
                               158096578/3914997-76204800/1304999*r78+
                               7257600/1304999*r67, 
                              -296462325/18269986+29030400/1304999*r78+
                               7257600/1304999*r68, 0, 0, 0, 0])


            block5 = np.array([-203718909/2096689+36288000/299527*r78+
                                7257600/299527*r67+29030400/299527*r68,
                                1477714693/3594324-152409600/299527*r78-
                                32659200/299527*r67-127008000/299527*r68,
                               -747867777/1198108+228614400/299527*r78+
                                54432000/299527*r67+203212800/299527*r68,
                                424854441/1198108-127008000/299527*r78-
                                36288000/299527*r67-127008000/299527*r68,
                                0,
                               -17380335/1198108+10886400/299527*r67+
                                25401600/299527*r68,
                               -67080435/1198108+25401600/299527*r78-
                                3628800/299527*r67, 
                                657798011/25160268-10886400/299527*r78-
                                3628800/299527*r68,- 2592/299527, 0, 0, 0])

            block6 = np.array([1529105/1237164- 
                               403200/103097*r67-1209600/103097*r68,
                              -3884117/618582+1935360/103097*r67+5644800/103097*r68, 
                               2611503/206194-3628800/103097*r67-10160640/103097*r68,
                              -7628371/618582+3225600/103097*r67+8467200/103097*r68, 
                               5793445/1237164-1209600/103097*r67-2822400/103097*r68,
                               0, 80640/103097*r67, 80640/103097*r68,
                               3072/103097, -288/103097, 0, 0 ])

            block7 = np.array([93255631/8041092-10886400/670091*r78+725760/670091*r67, 
                              -36069450/670091+50803200/670091*r78-3628800/670091*r67,
                               64346994/670091-91445760/670091*r78+7257600/670091*r67, 
                              -158096578/2010273+76204800/670091*r78
                              -7257600/670091*r67,
                               67080435/2680364-25401600/670091*r78+ 
                               3628800/670091*r67,
                              -725760/670091*r67, 0, 725760/670091*r78,
                              -145152/670091, 27648/670091,
                              -2592/670091, 0 ])

            block8 = np.array([-3921999/1079524+25401600/5127739*r78+
                                5080320/5127739*r68,
                                536324953/30766434-121927680/5127739*r78-
                                25401600/5127739*r68, 
                               -334792485/10255478+228614400/5127739*r78+
                                50803200/5127739*r68,
                                296462325/10255478-203212800/5127739*r78-
                                50803200/5127739*r68,
                               -657798011/61532868+76204800/5127739*r78+
                                25401600/5127739*r68,
                               -5080320/5127739*r68, 
                               -5080320/5127739*r78,
                                0, 4064256/5127739, -1016064/5127739,
                                193536/5127739, -18144/5127739])

            bd_block = np.array((block1,block2,block3,block4,
                                 block5,block6,block7,block8))/dx

            #stencil = np.array([,,,-4/5,0,4/5,-1/5,4/105,-1/280])
            q0 = 1/280
            q1 = -4/105
            q2 = 1/5
            q3 = -4/5
            q4 = 0
            q5 = -q3
            q6 = -q2
            q7 = -q1
            q8 = -q0

            D = sparse.diags([(Nx-4)*[q0], (Nx-3)*[q1], (Nx-2)*[q2],(Nx-1)*[q3], 
                              Nx*[q4], (Nx-1)*[q5],(Nx-2)*[q6],(Nx-3)*[q7],(Nx-4)*[q8]],
                              range(-4,5))/dx

            D = D.tolil()
            D[:8,:12] = bd_block
            D[-8:,-12:] = -np.rot90(D[:8,:12].todense(), k=2)

        self.P = sparse.diags([p], [0])
        self.P_inv = sparse.diags([p_inv], [0])

        if accuracy == 2 or accuracy == 4:
            self.D = self.P_inv@self.Q
        else:
            self.D     = D
            self.Q     = self.P@D 


class SBP2D:
    """ Class representing 2D finite difference SBP operators.

    This class defines 2D curvilinear SBP operators on a supplied grid X, Y,
    based on Ålund & Nordström (JCP, 2019).  Here X and Y are 2D numpy arrays
    representing the x- and y-values of the grid. X and Y should be structured
    such that (X[i,j], Y[i,j]) is equal to the (i,j):th grid node (x_ij, y_ij).

    Attributes:
        normals: A dictionary containing the normals for each boundary. The keys
            are 's' for south, 'e' for east, 'n' for north, 'w' for west.
            For example, normals['w']
        boundary_quadratures: A dictionary containing boundary quadratures for
            each boundary. I.e. arrays of weights that can be used to compute
            integrals over the boundaries.
        volume_quadrature: A matrix representing a quadrature over the domain.
    """

    def __init__(self, X, Y, accuracy = 2):
        """ Initializes an SBP2D object.

        Args:
            X: The x-values of the grid nodes.
            Y: The y-values of the grid nodes.

        Optional:
            accuracy: The accuracy of the interior stencils (2 or 4).

        """
        assert(X.shape == Y.shape)
        assert(accuracy in [2,4,8])

        self.X = X
        self.Y = Y
        (self.Nx, self.Ny) = X.shape

        self.Ix      = sparse.eye(self.Nx)
        self.Iy      = sparse.eye(self.Ny)
        self.sbp_xi  = SBP1D(self.Nx, 1/(self.Nx-1), accuracy)
        self.sbp_eta = SBP1D(self.Ny, 1/(self.Ny-1), accuracy)
        self.dx_dxi  = self.sbp_xi.D @ X
        self.dx_deta = X @ np.transpose(self.sbp_eta.D)
        self.dy_dxi  = self.sbp_xi.D @ Y
        self.dy_deta = Y @ np.transpose(self.sbp_eta.D)
        self.jac     = self.dx_dxi*self.dy_deta - self.dx_deta*self.dy_dxi
        self.sides   = { 'w': np.array([[x,y] for x,y in zip(X[0,:], Y[0,:])]),
                         'e': np.array([[x,y] for x,y in zip(X[-1,:], Y[-1,:])]),
                         's': np.array([[x,y] for x,y in zip(X[:,0], Y[:, 0])]),
                         'n': np.array([[x,y] for x,y in zip(X[:,-1], Y[:,-1])])}

        # Construct 2D SBP operators.
        self.J    = sparse.diags(self.jac.flatten())
        self.Jinv = sparse.diags(1/self.jac.flatten())
        self.Xxi  = sparse.diags(self.dx_dxi.flatten())
        self.Xeta = sparse.diags(self.dx_deta.flatten())
        self.Yxi  = sparse.diags(self.dy_dxi.flatten())
        self.Yeta = sparse.diags(self.dy_deta.flatten())
        self.Dxi  = sparse.kron(self.sbp_xi.D, self.Iy)
        self.Deta = sparse.kron(self.Ix, self.sbp_eta.D)
        self.Dx   = 0.5*self.Jinv*(self.Yeta @ self.Dxi +
                                   self.Dxi @ self.Yeta -
                                   self.Yxi @ self.Deta -
                                   self.Deta @ self.Yxi)
        self.Dy   = 0.5*self.Jinv*(self.Xxi @ self.Deta +
                                   self.Deta @ self.Xxi -
                                   self.Xeta @ self.Dxi -
                                   self.Dxi @ self.Xeta)
        self.P = self.J@sparse.kron(self.sbp_xi.P, self.sbp_eta.P)
        self.Pinv = sparse.diags(1/self.P.data)

        # Save matrix version of volume quadrature.
        self.volume_quadrature = np.reshape(self.P.diagonal(),
                                            (self.Nx, self.Ny))


        # Construct boundary quadratures.
        self.boundary_quadratures = {}
        self.pxi = np.diag(self.sbp_xi.P.todense())
        self.peta = np.diag(self.sbp_eta.P.todense())

        dx_deta_w = grid2d.get_function_boundary(self.dx_deta, 'w')
        dy_deta_w = grid2d.get_function_boundary(self.dy_deta, 'w')
        self.boundary_quadratures['w'] = \
                self.peta*np.sqrt(dx_deta_w**2 + dy_deta_w**2)

        dx_deta_e = grid2d.get_function_boundary(self.dx_deta, 'e')
        dy_deta_e = grid2d.get_function_boundary(self.dy_deta, 'e')
        self.boundary_quadratures['e'] = \
                self.peta*np.sqrt(dx_deta_e**2 + dy_deta_e**2)

        dx_dxi_s = grid2d.get_function_boundary(self.dx_dxi, 's')
        dy_dxi_s = grid2d.get_function_boundary(self.dy_dxi, 's')
        self.boundary_quadratures['s'] = \
                self.pxi*np.sqrt(dx_dxi_s**2 + dy_dxi_s**2)

        dx_dxi_n = grid2d.get_function_boundary(self.dx_dxi, 'n')
        dy_dxi_n = grid2d.get_function_boundary(self.dy_dxi, 'n')
        self.boundary_quadratures['n'] = \
                self.pxi*np.sqrt(dx_dxi_n**2 + dy_dxi_n**2)

        # Construct P^(-1) at boundaries.
        self.pinv = {}
        for side in ['s','e','n','w']:
            self.pinv[side] = 1/grid2d.get_function_boundary(
                    self.jac*np.outer(self.pxi, self.peta), side)

        # Compute normals.
        self.normals = {}
        self.normals['w'] = \
            np.array([ np.array([-nx, ny])/np.linalg.norm([nx, ny]) for
              (nx,ny) in zip(self.dy_deta[0,:], self.dx_deta[0,:]) ])
        self.normals['e'] = \
            np.array([ np.array([nx, -ny])/np.linalg.norm([nx, ny]) for
              (nx,ny) in zip(self.dy_deta[-1,:], self.dx_deta[-1,:]) ])
        self.normals['s'] = \
            np.array([ np.array([nx, -ny])/np.linalg.norm([nx, ny]) for
             (nx,ny) in zip(self.dy_dxi[:,0], self.dx_dxi[:,0]) ])
        self.normals['n'] = \
            np.array([ np.array([-nx, ny])/np.linalg.norm([nx, ny]) for
             (nx,ny) in zip(self.dy_dxi[:,-1], self.dx_dxi[:,-1]) ])


    def plot(self):
        """ Plots the grid and normals. """

        diam  = np.array([self.X[0,0]-self.X[-1,-1],self.Y[0,0]-self.Y[-1,-1]])
        scale = np.linalg.norm(diam) / np.max([self.Nx, self.Ny])

        fig, ax = plt.subplots()
        xmin    = np.min(self.X)
        xmax    = np.max(self.X)
        ymin    = np.min(self.Y)
        ymax    = np.max(self.Y)
        ax.set_xlim([xmin-scale,xmax+scale])
        ax.set_ylim([ymin-scale,ymax+scale])
        ax.plot(self.X, self.Y, 'b')
        ax.plot(np.transpose(self.X), np.transpose(self.Y), 'b')
        for side in ['w','e','s','n']:
            for p,n in zip(self.sides[side], self.normals[side]):
                ax.arrow(p[0], p[1], scale*n[0], scale*n[1],
                         head_width=0.01,
                         fc='k', ec='k')
        ax.axis('equal')
        plt.show()


    def diffx(self, u):
        """ Differentiates a grid function with respect to x. """
        return np.reshape(self.Dx@u.flatten(), (self.Nx, self.Ny))


    def diffy(self, u):
        """ Differentiates a grid function with respect to y. """
        return np.reshape(self.Dy@u.flatten(), (self.Nx, self.Ny))


    def integrate(self, u):
        """ Integrates a grid function over the domain. """
        return np.sum(self.P@u.flatten())
