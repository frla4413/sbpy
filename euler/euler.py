"""
Differential and SAT operators for Euler. A state should be thought of as a numpy array [U, V, P], where U, V, P are multiblock functions (i.e. lists of grid functions). The operators here operate on flattened versions of such a state. I.e. state = np.array([U, V, P]).flatten().

A complete spatial operator can be built by using the euler_operator function
together with boundary operators, for example, let sbp be a MultiblockSBP. Then:

def op(state):
    S,J = euler_operator(sbp, state) +
          wall_operator(sbp, state, 0, 'w') +
          wall_operator(sbp, state, 0, 'e') +
          wall_operator(sbp, state, 0, 's') +
          wall_operator(sbp, state, 0, 'n')
    return S,J

defines a spatial operator with wall conditions, which returns the operator and
its jacobian evaluated at the given state.

The system can then be integrated using the backward_euler function, or the
sbp_in_time function.
"""

import pdb
import warnings

import numpy as np
import scipy
from scipy import sparse
from tqdm import tqdm

from sbpy import operators
from sbpy.utils import export_to_tecplot


def vec_to_tensor(grid, vec):
    shapes = grid.get_shapes()
    component_length = np.prod([Nx*Ny for (Nx,Ny) in shapes])
    vec = np.reshape(vec, (4,component_length))

    start = 0
    U = []
    V = []
    W = []
    P = []
    for Nx,Ny in shapes:
        U.append(np.reshape(vec[0][start:(start+Nx*Ny)], (Nx,Ny)))
        V.append(np.reshape(vec[1][start:(start+Nx*Ny)], (Nx,Ny)))
        W.append(np.reshape(vec[2][start:(start+Nx*Ny)], (Nx,Ny)))
        P.append(np.reshape(vec[3][start:(start+Nx*Ny)], (Nx,Ny)))

    return np.array([U, V, W, P])

def euler_operator(sbp, state):
    """ The Euler spatial operator.
    Arguments:
        sbp - A MultilbockSBP object.
        state - A state vector

    Returns:
        S - The euler operator evaluated at the given state.
        J - The Jacobian of S at the given state.
    """

    u,v,w,p = vec_to_tensor(sbp.grid, state)

    r = sbp.grid.get_X(0)

    druudr = sbp.diffx(r*u*u)
    dudr = sbp.diffx(u)
    drpdr = sbp.diffx(r*p)
    dpdr = sbp.diffx(p)
    drwudz = sbp.diffy(r*w*u)
    dudz = sbp.diffy(u)

    drvudr = sbp.diffx(r*v*u)
    dvdr = sbp.diffx(v)
    drwvdz = sbp.diffy(r*w*v)
    dvdz = sbp.diffy(v)

    drwudr = sbp.diffx(r*w*u)
    dwdr = sbp.diffx(w)
    drwwdz = sbp.diffy(r*w*w)
    dwdz = sbp.diffy(w)
    drpdz = sbp.diffy(r*p)
    dpdz = sbp.diffy(p)

    drudr = sbp.diffx(r*u)
    drwdz = sbp.diffy(r*w)

    l1 = 0.5*(druudr + r*u*dudr + drpdr + r*dpdr + drwudz + r*w*dudz - p) - v*v
    l2 = 0.5*(drvudr + r*u*dvdr + drwvdz + r*w*dvdz) + u*v
    l3 = 0.5*(drwudr + r*u*dwdr + drwwdz + r*w*dwdz + drpdz + r*dpdz)
    l4 = 0.5*(drudr + r*dudr + drwdz + r*dwdz + u)

    #return np.array([l1,l2,l3,l4]).flatten()
    
    #Jacobian
    L = np.array([l1,l2,l3,l4]).flatten()
    Dr = sbp.get_Dx(0)
    Dz = sbp.get_Dy(0)
    U = sparse.diags(u[0].flatten())
    V = sparse.diags(v[0].flatten())
    W = sparse.diags(w[0].flatten())
    R = sparse.diags(r.flatten())
    I = sparse.eye(len(r.flatten()))
    Ur = sparse.diags(dudr.flatten())
    Uz = sparse.diags(dudz.flatten())
    Vr = sparse.diags(dvdr.flatten())
    Vz = sparse.diags(dvdz.flatten())
    Wr = sparse.diags(dwdr.flatten())
    Wz = sparse.diags(dwdz.flatten())

    dl1du = 0.5*(2*Dr@R@U + R@U@Dr + R@Ur + Dz@R@W + R@W@Dz)
    dl1dv = -2*V
    dl1dw = 0.5*(Dz@R@U + R@Uz)
    dl1dp = 0.5*(Dr@R + R@Dr - I)

    dl2du = 0.5*(Dr@R@V + R@Vr) + V
    dl2dv = 0.5*(Dr@R@U + R@U@Dr + Dz@R@W + R@W@Dz) + U
    dl2dw = 0.5*(Dz@R@V + R@Vz)
    dl2dp = None

    dl3du = 0.5*(Dr@R@W + R@Wr)
    dl3dv = None
    dl3dw = 0.5*(Dr@R@U + R@U@Dr + 2*Dz@R@W + R@W@Dz + R@Wz)
    dl3dp = 0.5*(Dz@R + R@Dz)

    dl4du = 0.5*(Dr@R + R@Dr + I)
    dl4dv = None
    dl4dw = 0.5*(Dz@R + R@Dz)
    dl4dp = None

    J = sparse.bmat([[dl1du, dl1dv, dl1dw, dl1dp],
                     [dl2du, dl2dv, dl2dw, dl2dp],
                     [dl3du, dl3dv, dl3dw, dl3dp],
                     [dl4du, dl4dv, dl4dw, dl4dp]])

    J = sparse.csr_matrix(J)

    return np.array([L, J], dtype=object)


def wall_operator(sbp, state, block_idx, side):
    u,v,w,_ = vec_to_tensor(sbp.grid, state)
    bd_slice = sbp.grid.get_boundary_slice(block_idx, side)
    u_bd = u[block_idx][bd_slice]
    v_bd = v[block_idx][bd_slice]
    w_bd = w[block_idx][bd_slice]
    n = sbp.get_normals(block_idx,side)
    nx = n[:,0]
    un   = nx*u_bd
    pinv = sbp.get_pinv(block_idx, side)
    bd_quad = sbp.get_boundary_quadrature(block_idx, side)
    lift = pinv*bd_quad

    r_bd = sbp.grid.get_X(block_idx)[bd_slice]

    Nx,Ny = sbp.grid.get_shapes()[0]
    num_blocks = sbp.grid.num_blocks
    s1 = np.zeros((num_blocks,Nx,Ny))
    s2 = np.zeros((num_blocks,Nx,Ny))
    s3 = np.zeros((num_blocks,Nx,Ny))
    s4 = np.zeros((num_blocks,Nx,Ny))

    s1[block_idx][bd_slice] = -0.5*lift*r_bd*u_bd*un
    s2[block_idx][bd_slice] = -0.5*lift*r_bd*v_bd*un
    s3[block_idx][bd_slice] = -0.5*lift*r_bd*w_bd*un
    s4[block_idx][bd_slice] = -lift*r_bd*un

    #return np.array([s1, s2, s3, s4]).flatten() 

    S = np.array([s1, s2, s3, s4]).flatten()

    #Jacobian
    ds1du = np.zeros((Nx,Ny))
    ds1du[bd_slice] = -0.5*lift*r_bd*(u_bd*nx + un)
    ds1du = sparse.diags(ds1du.flatten())
    ds1dv = np.zeros((Nx,Ny))
    ds1dv = sparse.diags(ds1dv.flatten())
    ds1dw = np.zeros((Nx,Ny))
    ds1dw = sparse.diags(ds1dw.flatten())
    ds1dp = np.zeros((Nx,Ny))
    ds1dp = sparse.diags(ds1dp.flatten())

    ds2du = np.zeros((Nx,Ny))
    ds2du[bd_slice] = -0.5*lift*r_bd*v_bd*nx
    ds2du = sparse.diags(ds2du.flatten())
    ds2dv = np.zeros((Nx,Ny))
    ds2dv[bd_slice] = -0.5*lift*r_bd*un
    ds2dv = sparse.diags(ds2dv.flatten())
    ds2dw = np.zeros((Nx,Ny))
    ds2dw = sparse.diags(ds2dw.flatten())
    ds2dp = np.zeros((Nx,Ny))
    ds2dp = sparse.diags(ds2dp.flatten())

    ds3du = np.zeros((Nx,Ny))
    ds3du[bd_slice] = -0.5*lift*r_bd*w_bd*nx
    ds3du = sparse.diags(ds3du.flatten())
    ds3dv = np.zeros((Nx,Ny))
    ds3dv = sparse.diags(ds3dv.flatten())
    ds3dw = np.zeros((Nx,Ny))
    ds3dw[bd_slice] = -0.5*lift*r_bd*un
    ds3dw = sparse.diags(ds3dw.flatten())
    ds3dp = np.zeros((Nx,Ny))
    ds3dp = sparse.diags(ds3dp.flatten())

    ds4du = np.zeros((Nx,Ny))
    ds4du[bd_slice] = -lift*r_bd*nx
    ds4du = sparse.diags(ds4du.flatten())
    ds4dv = np.zeros((Nx,Ny))
    ds4dv = sparse.diags(ds4dv.flatten())
    ds4dw = np.zeros((Nx,Ny))
    ds4dw = sparse.diags(ds4dw.flatten())
    ds4dp = np.zeros((Nx,Ny))
    ds4dp = sparse.diags(ds4dp.flatten())

    J = sparse.bmat([[ds1du, ds1dv, ds1dw, ds1dp,],
                     [ds2du, ds2dv, ds2dw, ds2dp],
                     [ds3du, ds3dv, ds3dw, ds3dp],
                     [ds4du, ds4dv, ds4dw, ds4dp]])

    return np.array([S,J], dtype=object)

def interior_penalty_operator_w(sbp, state, block_idx):
    "Penalty w = 0 at r = 0. Use only at west side."

    side = 'w'
    u,v,w,_ = vec_to_tensor(sbp.grid, state)
    bd_slice = sbp.grid.get_boundary_slice(block_idx, side)
    u_bd = u[block_idx][bd_slice]
    v_bd = v[block_idx][bd_slice]
    w_bd = w[block_idx][bd_slice]
    pinv = sbp.get_pinv(block_idx, side)

    Nx,Ny = sbp.grid.get_shapes()[0]
    num_blocks = sbp.grid.num_blocks
    s1 = np.zeros((num_blocks,Nx,Ny))
    s2 = np.zeros((num_blocks,Nx,Ny))
    s3 = np.zeros((num_blocks,Nx,Ny))
    s4 = np.zeros((num_blocks,Nx,Ny))

    s1[block_idx][bd_slice] = pinv*u_bd
    s2[block_idx][bd_slice] = pinv*v_bd

    #return np.array([s1, s2, s3, s4]).flatten() 

    S = np.array([s1, s2, s3, s4]).flatten()

    #Jacobian
    ds1du = np.zeros((Nx,Ny))
    ds1du[bd_slice] = pinv 
    ds1du = sparse.diags(ds1du.flatten())
    ds1dv = np.zeros((Nx,Ny))
    ds1dv = sparse.diags(ds1dv.flatten())
    ds1dw = np.zeros((Nx,Ny))
    ds1dw = sparse.diags(ds1dw.flatten())
    ds1dp = np.zeros((Nx,Ny))
    ds1dp = sparse.diags(ds1dp.flatten())

    ds2du = np.zeros((Nx,Ny))
    ds2du = sparse.diags(ds2du.flatten())
    ds2dv = np.zeros((Nx,Ny))
    ds2dv[bd_slice] = pinv 
    ds2dv = sparse.diags(ds2dv.flatten())
    ds2dw = np.zeros((Nx,Ny))
    ds2dw = sparse.diags(ds2dw.flatten())
    ds2dp = np.zeros((Nx,Ny))
    ds2dp = sparse.diags(ds2dp.flatten())

    ds3du = np.zeros((Nx,Ny))
    ds3du = sparse.diags(ds3du.flatten())
    ds3dv = np.zeros((Nx,Ny))
    ds3dv = sparse.diags(ds3dv.flatten())
    ds3dw = np.zeros((Nx,Ny))
    ds3dw = sparse.diags(ds3dw.flatten())
    ds3dp = np.zeros((Nx,Ny))
    ds3dp = sparse.diags(ds3dp.flatten())

    ds4du = np.zeros((Nx,Ny))
    ds4du = sparse.diags(ds4du.flatten())
    ds4dv = np.zeros((Nx,Ny))
    ds4dv = sparse.diags(ds4dv.flatten())
    ds4dw = np.zeros((Nx,Ny))
    ds4dw = sparse.diags(ds4dw.flatten())
    ds4dp = np.zeros((Nx,Ny))
    ds4dp = sparse.diags(ds4dp.flatten())

    J = sparse.bmat([[ds1du, ds1dv, ds1dw, ds1dp,],
                     [ds2du, ds2dv, ds2dw, ds2dp],
                     [ds3du, ds3dv, ds3dw, ds3dp],
                     [ds4du, ds4dv, ds4dw, ds4dp]])

    return np.array([S,J], dtype=object)


def outflow_operator(sbp, state, block_idx, side):

    ## for outflow boundary that might switch to inflow 

    u,v,w,p    = vec_to_tensor(sbp.grid, state)
    bd_slice = sbp.grid.get_boundary_slice(block_idx, side)
    normals = sbp.get_normals(block_idx, side)
    nx      = normals[:,0]
    ny      = normals[:,1]
    u_bd    = u[block_idx][bd_slice]
    v_bd    = v[block_idx][bd_slice]
    w_bd    = w[block_idx][bd_slice]
    p_bd    = p[block_idx][bd_slice]
    pinv    = sbp.get_pinv(block_idx, side)
    bd_quad = sbp.get_boundary_quadrature(block_idx, side)
    lift    = pinv*bd_quad

    r_bd = sbp.grid.get_X(block_idx)[bd_slice]

    Nx,Ny      = sbp.grid.get_shapes()[0]
    num_blocks = sbp.grid.num_blocks
    s1 = np.zeros((num_blocks,Nx,Ny))
    s2 = np.zeros((num_blocks,Nx,Ny))
    s3 = np.zeros((num_blocks,Nx,Ny))
    s4 = np.zeros((num_blocks,Nx,Ny))

    #parameters for sigmoid function, large a --> more steep curve, c --> shift sideways
    a = 15
    c = 0.2
    h = np.exp(c)/(np.exp(a*w_bd) + np.exp(c))

    u2 = u_bd**2 + v_bd**2 + w_bd**2
    s3[block_idx][bd_slice] = -lift*ny*r_bd*(h*u2+p_bd)

    #return np.array([s1, s2, s3, s4]).flatten()

    S = np.array([s1, s2, s3, s4]).flatten()

    #Jacobian
    dhdw = -a*np.exp(a*w_bd+c)/((np.exp(a*w_bd) + np.exp(c))**2)

    du2du = 2*u_bd # derivative of u2 w.r.t. u
    du2dv = 2*v_bd # derivative of u2 w.r.t. v
    du2dw = 2*w_bd # derivative of u2 w.r.t. w

    ds1du           = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds1dv           = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds1dw           = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds1dp           = sparse.csr_matrix((Nx*Ny, Nx*Ny))

    ds2du           = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds2dv           = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds2dw           = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds2dp           = sparse.csr_matrix((Nx*Ny, Nx*Ny))

    ds3du           = np.zeros((Nx,Ny))
    ds3du[bd_slice] = -lift*ny*r_bd*h*du2du
    ds3du           = sparse.diags(ds3du.flatten())

    ds3dv           = np.zeros((Nx,Ny))
    ds3dv[bd_slice] = -lift*ny*r_bd*h*du2dv
    ds3dv           = sparse.diags(ds3dv.flatten())

    ds3dw           = np.zeros((Nx,Ny))
    ds3dw[bd_slice] = -lift*ny*r_bd*(dhdw*u2+h*du2dw)
    ds3dw           = sparse.diags(ds3dw.flatten())
    ds3dv           = sparse.csr_matrix((Nx*Ny, Nx*Ny))

    ds3dp           = np.zeros((Nx,Ny))
    ds3dp[bd_slice] = -lift*ny*r_bd
    ds3dp           = sparse.diags(ds3dp.flatten())

    ds4du           = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds4dv           = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds4dw           = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds4dp           = sparse.csr_matrix((Nx*Ny, Nx*Ny))

    J = sparse.bmat([[ds1du, ds1dv, ds1dw, ds1dp],
                     [ds2du, ds2dv, ds2dw, ds2dp],
                     [ds3du, ds3dv, ds3dw, ds3dp],
                     [ds4du, ds4dv, ds4dw, ds4dp]])

    return np.array([S,J], dtype=object)


def backward_euler_step(op, prev_state, dt, tol, grid):
    N = int(len(prev_state)/4)

    R = grid.get_X(0).flatten()
    R_big = np.array([R, R, R, R]).flatten()

    def F(new_state):
        T = R_big*np.concatenate(
                [(new_state[0:int(3*N)] - prev_state[0:int(3*N)])/dt,
                 np.zeros(N)])
        S,Js = op(new_state)

        #Jacobian
        #I = sparse.identity(N)
        I = sparse.diags(R.flatten())
        O = sparse.csr_matrix((N,N))
        Jt = (1/dt)*sparse.bmat([[I, O, O, O],
                                 [O, I, O, O],
                                 [O, O, I, O],
                                 [O, O, O, O]])

        return T+S, Jt+Js

    new_state = prev_state.copy()
    n_iter = 0
    while True:
        L, J = F(new_state)
        err = np.linalg.norm(L, ord=np.inf)
        print(err)
        
        if err < tol:
            break

        if err > 1e7 or n_iter > 20: #temp magic constant
            raise Exception("Newton diverging, try decreasing dt.")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
#                delta = sparse.linalg.spsolve(J,L)
                delta,exitCode = sparse.linalg.gmres(J,L, atol = tol)
                print(exitCode)
        except sparse.linalg.MatrixRankWarning:
            #Small perturbation if singular Jacobian
            delta = np.random.normal(scale=1e-7, size=len(prev_state))

        new_state -= delta
        n_iter += 1

    #print("Error: {:.2e}".format(err))
    return new_state

def backward_euler_step_newton_krylov(op, prev_state, dt, tol, grid):
    N = int(len(prev_state)/4)

    X = grid.get_X(0).flatten()
    X_big = np.array([X, X, X, X]).flatten()

    def F(new_state):
        T = np.concatenate(
                [(new_state[0:int(3*N)] - prev_state[0:int(3*N)]),
                 np.zeros(N)])
        S = op(new_state)
        return X_big*T+S*dt

    n_iter = 0
    err = 10
    new_state = prev_state.copy()

    while err > tol and n_iter < 5:
        new_state = scipy.optimize.newton_krylov(F,new_state, verbose=True,
                                                 iter = 200, f_tol=tol) 
        err = np.max(np.abs(F(new_state)))
        n_iter += 1
    return new_state


def solve(grid, spatial_operator, init_u, init_v, init_w, init_p, dt, num_steps,
          name_base):
    """ Solves an Euler problem.
    Arguments:
        grid: A MultiblockGrid object associated to the problem.
        spatial_operator: A spatial operator built from the euler_operator 
            plus SAT operators.
        init_u: Initial data for u as a multiblock function.
        init_v: Initial data for v as a multiblock function.
        init_p: Initial data for p as a multiblock function.

    Returns:
        U: A list of multiblock functions representing u in each time step.
        V: A list of multiblock functions representing v in each time step.
        P: A list of multiblock functions representing p in each time step.
    """
    P=[]
    U=[]
    V=[]
    W=[]
    sol = np.array([init_u, init_v, init_w, init_p]).flatten()
    sol = vec_to_tensor(grid, sol)
    U.append(sol[0][0])
    V.append(sol[1][0])
    W.append(sol[2][0])
    P.append(sol[3][0])
    sol = sol.flatten()

    spat_op = lambda x: spatial_operator(x)[0]
    tol = 1e-10
    for k in tqdm(range(num_steps)):
        #sol = backward_euler_step_newton_krylov(spat_op, sol, dt, 
        #                                                       tol, grid)
        sol = backward_euler_step(spatial_operator, sol, dt, tol, grid)

        sol = vec_to_tensor(grid, sol)
        U.append(sol[0][0])
        V.append(sol[1][0])
        W.append(sol[2][0])
        P.append(sol[3][0])
        filename = name_base+str(k)+'.tec'
        export_to_tecplot(grid,sol[0][0], sol[1][0], sol[2][0],sol[3][0],filename)
        sol = sol.flatten()

    return U,V,W,P
