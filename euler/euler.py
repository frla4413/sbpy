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


def vec_to_tensor(grid, vec):
    shapes = grid.get_shapes()
    component_length = np.prod([Nx*Ny for (Nx,Ny) in shapes])
    vec = np.reshape(vec, (3,component_length))

    start = 0
    U = []
    V = []
    P = []
    for Nx,Ny in shapes:
        U.append(np.reshape(vec[0][start:(start+Nx*Ny)], (Nx,Ny)))
        V.append(np.reshape(vec[1][start:(start+Nx*Ny)], (Nx,Ny)))
        P.append(np.reshape(vec[2][start:(start+Nx*Ny)], (Nx,Ny)))

    return np.array([U, V, P])

def euler_operator(sbp, state):
    """ The Euler spatial operator.
    Arguments:
        sbp - A MultilbockSBP object.
        state - A state vector

    Returns:
        S - The euler operator evaluated at the given state.
        J - The Jacobian of S at the given state.
    """

    u,v,p = vec_to_tensor(sbp.grid, state)

    dudx = sbp.diffx(u)
    dudy = sbp.diffy(u)
    duudx = sbp.diffx(u*u)
    dvdx = sbp.diffx(v)
    dvdy = sbp.diffy(v)
    dvvdy = sbp.diffy(v*v)
    duvdx = sbp.diffx(u*v)
    duvdy = sbp.diffy(u*v)
    dpdx = sbp.diffx(p)
    dpdy = sbp.diffy(p)

    l1 = 0.5*(u*dudx + v*dudy + duudx + duvdy) + dpdx
    l2 = 0.5*(u*dvdx + v*dvdy + dvvdy + duvdx) + dpdy
    l3 = dudx + dvdy

    L = np.array([l1,l2,l3]).flatten()

    #Jacobian
    Dx = sbp.get_Dx(0)
    Dy = sbp.get_Dy(0)
    U = sparse.diags(u[0].flatten())
    V = sparse.diags(v[0].flatten())
    Ux = sparse.diags(dudx.flatten())
    Uy = sparse.diags(dudy.flatten())
    Vx = sparse.diags(dvdx.flatten())
    Vy = sparse.diags(dvdy.flatten())

    dl1du = 0.5*(U@Dx + Ux + V@Dy + 2*Dx@U + Dy@V)
    dl1dv = 0.5*(Uy + Dy@U)
    dl1dp = Dx
    dl2du = 0.5*(Vx + Dx@V)
    dl2dv = 0.5*(U@Dx + V@Dy + Vy + Dx@U + 2*Dy@V)
    dl2dp = Dy
    dl3du = Dx
    dl3dv = Dy
    dl3dp = None

    J = sparse.bmat([[dl1du, dl1dv, dl1dp],
                     [dl2du, dl2dv, dl2dp],
                     [dl3du, dl3dv, dl3dp]])

    J = sparse.csr_matrix(J)

    return np.array([L, J], dtype=object)


def wall_operator(sbp, state, block_idx, side):
    u,v,p = vec_to_tensor(sbp.grid, state)
    bd_slice = sbp.grid.get_boundary_slice(block_idx, side)
    normals = sbp.get_normals(block_idx, side)
    nx = normals[:,0]
    ny = normals[:,1]
    u_bd = u[block_idx][bd_slice]
    v_bd = v[block_idx][bd_slice]
    wn = u_bd*nx + v_bd*ny
    pinv = sbp.get_pinv(block_idx, side)
    bd_quad = sbp.get_boundary_quadrature(block_idx, side)
    lift = pinv*bd_quad

    Nx,Ny = sbp.grid.get_shapes()[0]
    num_blocks = sbp.grid.num_blocks
    s1 = np.zeros((num_blocks,Nx,Ny))
    s2 = np.zeros((num_blocks,Nx,Ny))
    s3 = np.zeros((num_blocks,Nx,Ny))

    s1[block_idx][bd_slice] = -0.5*lift*u_bd*wn
    s2[block_idx][bd_slice] = -0.5*lift*v_bd*wn
    s3[block_idx][bd_slice] = -lift*wn

    S = np.array([s1, s2, s3]).flatten()

    #Jacobian
    ds1du = np.zeros((Nx,Ny))
    ds1du[bd_slice] = 0.5*lift*(u_bd*nx + wn)
    ds1du = sparse.diags(ds1du.flatten())
    ds1dv = np.zeros((Nx,Ny))
    ds1dv[bd_slice] = 0.5*lift*u_bd*ny
    ds1dv = sparse.diags(ds1dv.flatten())
    ds1dp = np.zeros((Nx,Ny))
    ds1dp = sparse.diags(ds1dp.flatten())

    ds2du = np.zeros((Nx,Ny))
    ds2du[bd_slice] = 0.5*lift*v_bd*nx
    ds2du = sparse.diags(ds2du.flatten())
    ds2dv = np.zeros((Nx,Ny))
    ds2dv[bd_slice] = 0.5*lift*(v_bd*ny + wn)
    ds2dv = sparse.diags(ds2dv.flatten())
    ds2dp = np.zeros((Nx,Ny))
    ds2dp = sparse.diags(ds2dp.flatten())

    ds3du = np.zeros((Nx,Ny))
    ds3du[bd_slice] = lift*nx
    ds3du = sparse.diags(ds3du.flatten())
    ds3dv = np.zeros((Nx,Ny))
    ds3dv[bd_slice] = lift*ny
    ds3dv = sparse.diags(ds3dv.flatten())
    ds3dp = np.zeros((Nx,Ny))
    ds3dp = sparse.diags(ds3dp.flatten())


    J = -sparse.bmat([[ds1du, ds1dv, ds1dp],
                      [ds2du, ds2dv, ds2dp],
                      [ds3du, ds3dv, ds3dp]])

    return np.array([S,J], dtype=object)


def inflow_operator(sbp, state, block_idx, side, wn_data, wt_data):
    u,v,p = vec_to_tensor(sbp.grid, state)
    bd_slice = sbp.grid.get_boundary_slice(block_idx, side)
    normals = sbp.get_normals(block_idx, side)
    nx = normals[:,0]
    ny = normals[:,1]
    u_bd = u[block_idx][bd_slice]
    v_bd = v[block_idx][bd_slice]
    wn = u_bd*nx + v_bd*ny
    wt = -u_bd*ny + v_bd*nx
    pinv = sbp.get_pinv(block_idx, side)
    bd_quad = sbp.get_boundary_quadrature(block_idx, side)
    lift = pinv*bd_quad

    Nx,Ny = sbp.grid.get_shapes()[0]
    num_blocks = sbp.grid.num_blocks
    s1 = np.zeros((num_blocks,Nx,Ny))
    s2 = np.zeros((num_blocks,Nx,Ny))
    s3 = np.zeros((num_blocks,Nx,Ny))

    s1[block_idx][bd_slice] = -lift*nx*wn*(wn-wn_data) \
                              +lift*ny*wn*(wt-wt_data)
    s2[block_idx][bd_slice] = -lift*ny*wn*(wn-wn_data) \
                              -lift*nx*wn*(wt-wt_data)
    s3[block_idx][bd_slice] = -lift*(wn-wn_data)

    S = np.array([s1, s2, s3]).flatten()

    #Jacobian
    ds1du = np.zeros((Nx,Ny))
    ds1du[bd_slice] = -lift*(nx*nx*(2*wn-wn_data) + ny*ny*wn - nx*ny*(wt-wt_data))
    ds1du = sparse.diags(ds1du.flatten())
    ds1dv = np.zeros((Nx,Ny))
    ds1dv[bd_slice] = -lift*(nx*ny*(2*wn-wn_data) - ny*nx*wn - ny*ny*(wt-wt_data))
    ds1dv = sparse.diags(ds1dv.flatten())
    ds1dp = np.zeros((Nx,Ny))
    ds1dp = sparse.diags(ds1dp.flatten())

    ds2du = np.zeros((Nx,Ny))
    ds2du[bd_slice] = -lift*(nx*ny*(2*wn-wn_data) - nx*ny*wn + nx*nx*(wt-wt_data))
    ds2du = sparse.diags(ds2du.flatten())
    ds2dv = np.zeros((Nx,Ny))
    ds2dv[bd_slice] = -lift*(ny*ny*(2*wn-wn_data) + nx*nx*wn + nx*ny*(wt-wt_data))
    ds2dv = sparse.diags(ds2dv.flatten())
    ds2dp = np.zeros((Nx,Ny))
    ds2dp = sparse.diags(ds2dp.flatten())

    ds3du = np.zeros((Nx,Ny))
    ds3du[bd_slice] = -lift*nx
    ds3du = sparse.diags(ds3du.flatten())
    ds3dv = np.zeros((Nx,Ny))
    ds3dv[bd_slice] = -lift*ny
    ds3dv = sparse.diags(ds3dv.flatten())
    ds3dp = np.zeros((Nx,Ny))
    ds3dp = sparse.diags(ds3dp.flatten())


    J = sparse.bmat([[ds1du, ds1dv, ds1dp],
                     [ds2du, ds2dv, ds2dp],
                     [ds3du, ds3dv, ds3dp]])

    return np.array([S,J], dtype=object)


def pressure_operator(sbp, state, block_idx, side):
    u,v,p = vec_to_tensor(sbp.grid, state)
    bd_slice = sbp.grid.get_boundary_slice(block_idx, side)
    normals = sbp.get_normals(block_idx, side)
    nx = normals[:,0]
    ny = normals[:,1]
    p_bd = p[block_idx][bd_slice]

    pinv = sbp.get_pinv(block_idx, side)
    bd_quad = sbp.get_boundary_quadrature(block_idx, side)
    lift = pinv*bd_quad

    Nx,Ny = sbp.grid.get_shapes()[0]
    num_blocks = sbp.grid.num_blocks
    s1 = np.zeros((num_blocks,Nx,Ny))
    s1[block_idx][bd_slice] = -lift*nx*p_bd
    s2 = np.zeros((num_blocks,Nx,Ny))
    s2[block_idx][bd_slice] = -lift*ny*p_bd
    s3 = np.zeros((num_blocks,Nx,Ny))

    S = np.array([s1, s2, s3]).flatten()

    #Jacobian
    ds1du = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds1dv = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds1dp = np.zeros((Nx,Ny))
    ds1dp[bd_slice] = lift*nx
    ds1dp = sparse.diags(ds1dp.flatten())

    ds2du = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds2dv = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds2dp = np.zeros((Nx,Ny))
    ds2dp[bd_slice] = lift*ny
    ds2dp = sparse.diags(ds2dp.flatten())

    ds3du = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds3dv = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds3dp = sparse.csr_matrix((Nx*Ny, Nx*Ny))


    J = -sparse.bmat([[ds1du, ds1dv, ds1dp],
                      [ds2du, ds2dv, ds2dp],
                      [ds3du, ds3dv, ds3dp]])

    return np.array([S,J], dtype=object)


def outflow_operator(sbp, state, block_idx, side):

    ## for outflow boundary that might switch to inflow 

    u,v,p    = vec_to_tensor(sbp.grid, state)
    bd_slice = sbp.grid.get_boundary_slice(block_idx, side)
    normals = sbp.get_normals(block_idx, side)
    nx      = normals[:,0]
    ny      = normals[:,1]
    u_bd    = u[block_idx][bd_slice]
    v_bd    = v[block_idx][bd_slice]
    p_bd    = p[block_idx][bd_slice]
    wn      = u_bd*nx + v_bd*ny
    wt      = -u_bd*ny + v_bd*nx
    pinv    = sbp.get_pinv(block_idx, side)
    bd_quad = sbp.get_boundary_quadrature(block_idx, side)
    lift    = pinv*bd_quad

    Nx,Ny      = sbp.grid.get_shapes()[0]
    num_blocks = sbp.grid.num_blocks
    s1 = np.zeros((num_blocks,Nx,Ny))
    s2 = np.zeros((num_blocks,Nx,Ny))
    s3 = np.zeros((num_blocks,Nx,Ny))

    #parameters for sigmoid function, large a --> more steep curve, c --> shift sideways
    a = 15
    c = 0.2
    h = np.exp(c)/(np.exp(a*wn) + np.exp(c))

    w2 = wn*wn+wt*wt
    s1[block_idx][bd_slice] = -lift*nx*(h*w2+p_bd)
    s2[block_idx][bd_slice] = -lift*ny*(h*w2+p_bd)

    S = np.array([s1, s2, s3]).flatten()

    #Jacobian
    dhdwn = -a*np.exp(a*wn+c)/((np.exp(a*wn) + np.exp(c))**2)
    dhdu  = dhdwn*nx
    dhdv  = dhdwn*ny

    dw2du = 2*(wn*nx-wt*ny) # derivative of wn^2+wt^2 w.r.t. u
    dw2dv = 2*(wn*ny+wt*nx) # derivative of wn^2+wt^2 w.r.t. v

    ds1du           = np.zeros((Nx,Ny))
    ds1du[bd_slice] = -lift*nx*(dhdu*w2+h*dw2du)
    ds1du           = sparse.diags(ds1du.flatten())

    ds1dv           = np.zeros((Nx,Ny))
    ds1dv[bd_slice] = -lift*nx*(dhdv*w2+h*dw2dv)
    ds1dv           = sparse.diags(ds1dv.flatten())

    ds1dp           = np.zeros((Nx,Ny))
    ds1dp[bd_slice] = -lift*nx
    ds1dp           = sparse.diags(ds1dp.flatten())

    ds2du           = np.zeros((Nx,Ny))
    ds2du[bd_slice] = -lift*ny*(dhdu*w2+h*dw2du)
    ds2du           = sparse.diags(ds2du.flatten())

    ds2dv           = np.zeros((Nx,Ny))
    ds2dv[bd_slice] = -lift*ny*(dhdv*w2+h*dw2dv)
    ds2dv           = sparse.diags(ds2dv.flatten())

    ds2dp           = np.zeros((Nx,Ny))
    ds2dp[bd_slice] = -lift*ny
    ds2dp           = sparse.diags(ds2dp.flatten())

    ds3du           = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds3dv           = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds3dp           = sparse.csr_matrix((Nx*Ny, Nx*Ny))

    J = sparse.bmat([[ds1du, ds1dv, ds1dp],
                     [ds2du, ds2dv, ds2dp],
                     [ds3du, ds3dv, ds3dp]])

    return np.array([S,J], dtype=object)


def bdf1_step(op, prev_state, dt, tol):
    N = int(len(prev_state)/3)

    def F(new_state, prev_state):
        T = np.concatenate(
                [(new_state[0:int(2*N)] - prev_state[0:int(2*N)]),
                 np.zeros(N)])
        S,Js = op(new_state)

        #Jacobian
        I = sparse.identity(N)
        O = sparse.csr_matrix((N,N))
        Jt = sparse.bmat([[I, O, O],
                          [O, I, O],
                          [O, O, O]])

        return T+dt*S, Jt+dt*Js

    new_state = prev_state.copy()
    n_iter = 0
    while True:
        L, J = F(new_state, prev_state)
        err = np.linalg.norm(L, ord=np.inf)
        if err < tol:
            break

        if err > 1000 or n_iter > 10: #temp magic constant
            raise Exception("Newton diverging, try decreasing dt.")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                delta = sparse.linalg.spsolve(J,L)
        except sparse.linalg.MatrixRankWarning:
            #Small perturbation if singular Jacobian
            delta = np.random.normal(scale=1e-7, size=len(prev_state))

        new_state -= delta
        n_iter += 1

    #print("Error: {:.2e}".format(err))
    return new_state

def bdf2_step(op, prev_state, prev_prev_state, dt, tol):
    N = int(len(prev_state)/3)

    def F(new_state, prev_state):
        T = np.concatenate(
           [(new_state[0:int(2*N)]
            -(4/3)*prev_state[0:int(2*N)]
            + (1/3)*prev_prev_state[0:int(2*N)]),
           np.zeros(N)])
        S,Js = op(new_state)

        #Jacobian
        I = sparse.identity(N)
        O = sparse.csr_matrix((N,N))
        Jt = sparse.bmat([[I, O, O],
                          [O, I, O],
                          [O, O, O]])

        return T+(2*dt/3)*S, Jt+(2*dt/3)*Js

    new_state = prev_state.copy()
    n_iter = 0
    while True:
        L, J = F(new_state, prev_state)
        err = np.linalg.norm(L, ord=np.inf)
        if err < tol:
            break

        if err > 1000 or n_iter > 10: #temp magic constant
            raise Exception("Newton diverging, try decreasing dt.")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                delta = sparse.linalg.spsolve(J,L)
        except sparse.linalg.MatrixRankWarning:
            #Small perturbation if singular Jacobian
            delta = np.random.normal(scale=1e-7, size=len(prev_state))

        new_state -= delta
        n_iter += 1

    #print("Error: {:.2e}".format(err))
    return new_state



def solve(grid, spatial_operator, init_u, init_v, init_p, dt, num_steps):
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
    sol = np.array([init_u, init_v, init_p]).flatten()
    tol = 1e-12
    for k in tqdm(range(num_steps)):
        prev_state = sol
        if k == 0:
            sol = bdf1_step(spatial_operator, prev_state, dt, tol)
        else:
            prev_prev_state = prev_state
            prev_state = sol
            sol = bdf2_step(spatial_operator, prev_state, prev_prev_state, dt, tol)


        sol = vec_to_tensor(grid, sol)
        U.append(sol[0][0])
        V.append(sol[1][0])
        P.append(sol[2][0])
        sol = sol.flatten()

    return U,V,P



def backward_euler_step_newton_krylov(op, prev_state, dt, tol):
    N = int(len(prev_state)/3)

    def F(new_state):
        T = np.concatenate(
                [(new_state[0:int(2*N)] - prev_state[0:int(2*N)])/dt,
                 np.zeros(N)])
        S = op(new_state)[0]

        return T+S

    n_iter = 0
    err = 10
    new_state = prev_state.copy()

    while err > tol and n_iter < 5:
        new_state = scipy.optimize.newton_krylov(F,prev_state, verbose=False,
                                                 maxiter = 200, f_tol=1e-12) 
        err = np.max(np.abs(F(new_state)))
        n_iter += 1
    return new_state

def sbp_in_time_step(op, cur_state, dt, tol):
    N = int(len(cur_state)/3)

    def F(prev_state, next_state):
        T = np.concatenate([(next_state[0:2*N] - 0.75*prev_state[0:2*N] - 0.25*cur_state[0:2*N])/dt,
                     np.zeros(N),
                     (next_state[0:2*N] - prev_state[0:2*N])/dt,
                     np.zeros(N)])


        S0,Js0 = op(prev_state)
        S1,Js1 = op(next_state)
        S = np.concatenate([S0, S1])

        #Jacobian
        I = sparse.identity(N)
        O = sparse.csr_matrix((N,N))
        Jt = (1/dt)*sparse.bmat([[-0.75*I,       O, O, I, O, O],
                                 [      O, -0.75*I, O, O, I, O],
                                 [      O,       O, O, O, O, O],
                                 [     -I,       O, O, I, O, O],
                                 [      O,      -I, O, O, I, O],
                                 [      O,       O, O, O, O, O]])

        J = Jt + sparse.bmat([[Js0, None],
                              [None, Js1]])

        return T+S,J

    prev_state = cur_state.copy()
    new_state = cur_state.copy()
    while True:
        L, J = F(prev_state, new_state)
        err = np.linalg.norm(L, ord=np.inf)
        if err < tol:
            break

        delta = sparse.linalg.spsolve(J,L)
        prev_state -= delta[0:3*N]
        new_state -= delta[3*N:]

    #print("Error: {:.2e}".format(err))
    return new_state

def solve_steady_state(grid, spatial_operator, init_u, init_v,init_p):
    """ Solves the steady-state version of an Euler problem.
    Arguments:
        grid: A MultiblockGrid object associated to the problem.
        spatial_operator: A spatial operator built from the euler_operator 
            plus SAT operators.
        init_u: Initial guess for u as a multiblock function.
        init_v: Initial guess for v as a multiblock function.
        init_p: Initial guess for p as a multiblock function.

    Returns:
        U: A list of multiblock functions representing u in each time step.
        V: A list of multiblock functions representing v in each time step.
        P: A list of multiblock functions representing p in each time step.
    """
    P=[]
    U=[]
    V=[]
    sol = np.array([init_u, init_v, init_p]).flatten()
    max_iter = 50
    tol      = 1e-12
    for k in range(max_iter):

        F,J = spatial_operator(sol)
        err = np.linalg.norm(F,ord=np.inf) 

        if err < 1e-1:
            alpha = 1

        if err < tol:
            break

        #msg = 'error = ' + err + ' at iteration ' + k
        print("error: ", err)

        sol-= sparse.linalg.spsolve(J,F)
        sol = vec_to_tensor(grid, sol)
        U.append(sol[0][0])
        V.append(sol[1][0])
        P.append(sol[2][0])
        sol = sol.flatten()

    print("Converged in ", k, "iterations")
    err = []

    # print errors for convergence rate in Newton steps
    for k in range(len(U)):
        err.append(np.linalg.norm(U[-1]-U[k],ord=np.inf))

    print("Error in inf-norm, u^* is the last iterate")
    print(err)
    return U,V,P
