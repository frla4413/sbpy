"""This module contains functions for getting DG-SBP operators."""

import pdb

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from sbpy import grid2d
import math

def almost_equal(a,b):
    eps = np.finfo(float).eps
    
    if a == 0 or b == 0: 
        if abs(a-b) <= 2*eps:
            return True
        else:
            return False
    else:
        if abs(a-b) <= eps*abs(a) and abs(a-b) <= eps*abs(b):
            return True
        else:
            return False

def barycentric_weights(nodes):
    w = np.ones(len(nodes))

    for j in range(1,len(w)):
        for k in range(j):
            w[k] = w[k]*(nodes[k] - nodes[j])
            w[j] = w[j]*(nodes[j] - nodes[k])
    w = 1/w
    return w

def lagrange_interpolation(x,nodes,f,w):
    N = len(nodes)
    numerator   = 0
    denominator = 0

    for j in range(N):
        if almost_equal(x,nodes[j]):
            return f[j]

        t            = w[j]/(x-nodes[j])
        numerator   += t*f[j]
        denominator +=t

    return numerator/denominator

def lagrange_interpolation_derivative(x, nodes, f, w):
    at_node = False
    num     = 0
    N       = len(nodes)
    for j in range(N):
        if almost_equal(x, nodes[j]): 
            at_node = True
            p       = f[j]
            denum   = -w[j]
            i       = j

    if at_node:
        for j in range(N):
            if j != i: 
                num += w[j]*(p-f[j])/(x - nodes[j])
    else:
        denum = 0
        p = lagrange_interpolation(x, nodes, f, w)

        for j in range(N):
            t       = w[j]/(x - nodes[j])
            num    += t*(p - f[j])/(x - nodes[j])
            denum  += t
    return num/denum #p_prime

def build_interpolant(x,nodes,f,w):
    return [ lagrange_interpolation(j,nodes,f,w) for j in x]


def polynomial_derivative_matrix(nodes):
    w = barycentric_weights(nodes)
    N = len(nodes)
    D = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i,j] = w[j]/(w[i]*(nodes[i]-nodes[j]))
                D[i,i]-= D[i,j]
    return D

def q_and_l_evaluation(N,x):
    assert N >= 2
    L_N_m_2 = 1
    L_N_m_1 = x

    L_prime_N_m_2  = 0
    L_prime_N_m_1  = 1

    for k in range(2,N+1):
        L_N           = ((2*k-1)*x*L_N_m_1 - (k-1)*L_N_m_2)/k
        L_prime_N     = L_prime_N_m_2 + (2*k-1)*L_N_m_1
        L_N_m_2       = L_N_m_1
        L_N_m_1       = L_N
        L_prime_N_m_2 = L_prime_N_m_1
        L_prime_N_m_1 = L_prime_N

    k               = N+1
    L_N_p_1         = ((2*k-1)*x*L_N - (k-1)*L_N_m_2)/k
    L_prime_N_p_1   = L_prime_N_m_2 + (2*k-1)*L_N_m_1
    q               = L_N_p_1 - L_N_m_2
    q_prime         = L_prime_N_p_1 - L_prime_N_m_2
    #pdb.set_trace()

    return q,q_prime,L_N

def legendre_gauss_lobatto_nodes_and_weights(N):
    nodes = np.zeros(N+1)
    w     = np.zeros(N+1)
    max_it=5

    tol   = 4*np.finfo(float).eps
    if N == 1:
        nodes = np.array((-1,1))
        w     = np.array([1,1])
        return nodes,w
    else: 
        nodes[0] = -1
        nodes[-1]= 1
        w[0] = 2/(N*(N+1))
        w[-1]= w[0]
        for j in range(1,math.floor((N+1)/2)):
            arg = ((j+1/4)*math.pi)/N - 3/((8*N*math.pi)*(j+1/4))
            nodes[j] = -math.cos(arg)
            for k in range(max_it):
                q,q_prime,L_N = q_and_l_evaluation(N,nodes[j])
                delta = -q/q_prime
                nodes[j]+= delta
                if abs(delta) <= tol*abs(nodes[j]):
                    break
                q,q_prime,L_N = q_and_l_evaluation(N,nodes[j])
                nodes[N-j] = -nodes[j]
                w[j] = 2/(N*(N+1)*L_N**2)
                w[N-j] = w[j]
    if N % 2 == 0:
        q,q_prime,L_N = q_and_l_evaluation(N,0)
        nodes[int(N/2)] = 0
        w[int(N/2)]     = 2/(N*(N+1)*L_N**2)

    return nodes, w

def transform_to_unit_interval(a,b,nodes_vals):
        return 2*(nodes_vals - (b+a))/(b-a)

class CurveInterpolant:
    """ A curve interpolant class. 
        From Kopriva, Algorithm 96
    """

    def __init__(self, nodes, x, y): 

        self.N        = len(nodes)
        self.nodes    = nodes
        self.x        = x
        self.y        = y
        self.weights  = barycentric_weights(nodes)


    def evaluate_at(self,s):
        x = lagrange_interpolation(s, self.nodes, self.x, self.weights)
        y = lagrange_interpolation(s, self.nodes, self.y, self.weights)
        return x,y

    def derivative_at(self,s):
        x_prime = lagrange_interpolant_derivative(s, self.nodes, self.x, self.weights)
        y_prime = lagrange_interpolant_derivative(s, self.nodes, self.x, self.weights)
        return x_prime, y_prime

    def get_N(self):
        return self.N

class DGSBP1D:
    """ Class representing a 1D DG-SBP operator.
        The algorithms are taken from "Implementing Spectal Methods".

    Attributes:
        P: Quadrature matrix.
        Q: An almost skew-symmetric matrix (Q+Q^T = diag(-1,0,0,...,1)), such
            that P^(-1)Q is an SBP operator.
        D: The SBP operator P^(-1)Q.
    """

    def __init__(self, N):
        """ Initializes an SBP1D object.

        Args:
            N: The number of grid points.
            The operator differentiates polynomials up to order N-1 exact
        """

        self.N        = N
        nodes,weights = legendre_gauss_lobatto_nodes_and_weights(N-1)
        self.D        = polynomial_derivative_matrix(nodes)
        self.P        = sparse.diags([weights], [0])
        self.P_inv    = sparse.diags([1/weights], [0])
        self.Q        = self.P@self.D

class DGSBP2D:
    """ Class representing 2D DG-SBP operators.

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

    def __init__(self, X, Y):
        """ Initializes an SBP2D object.

        Args:
            X: The x-values of the grid nodes.
            Y: The y-values of the grid nodes.

        Accuracy: In contrast to FD-operators, the accuracy is determined by the number
                  of points.
        """
        assert(X.shape == Y.shape)

        self.X = X
        self.Y = Y
        (self.Nx, self.Ny) = X.shape

        self.Ix      = sparse.eye(self.Nx)
        self.Iy      = sparse.eye(self.Ny)
        self.sbp_xi  = DGSBP1D(self.Nx)
        self.sbp_eta = DGSBP1D(self.Ny)
        self.dx_dxi  = self.sbp_xi.D @ X
        self.dx_deta = X @ np.transpose(self.sbp_eta.D)
        self.dy_dxi  = self.sbp_xi.D @ Y
        self.dy_deta = Y @ np.transpose(self.sbp_eta.D)
        self.jac     = self.dx_dxi*self.dy_deta - self.dx_deta*self.dy_dxi
        self.sides   = { 'w': np.array([[x,y] for x,y in zip(X[0,:], Y[0,:])]),
                         'e': np.array([[x,y] for x,y in zip(X[-1,:], Y[-1,:])]),
                         's': np.array([[x,y] for x,y in zip(X[:,0], Y[:, 0])]),
                         'n': np.array([[x,y] for x,y in zip(X[:,-1], Y[:,-1])])}

        # Construct 2D DGSBP operators.
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
