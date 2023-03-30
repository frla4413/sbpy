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
import mms_abl as mms
from scipy.sparse.linalg import LinearOperator

def vec_to_tensor(grid, vec):
    shapes = grid.get_shapes()
    component_length = np.sum([Nx*Ny for (Nx,Ny) in shapes])

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

def grid_function_to_diagonal_matrix(F):
    F_mat = F.reshape((F.shape[0]*F.shape[1],1),order = 'F')
    F_mat = F_mat.toarray()
    return sparse.diags(F_mat[:,0])


def force_operator(sbp,force1_func,force2_func,force3_func,t):

    force1 = []
    force2 = []
    force3 = []

    for i in range(len(sbp.grid.get_blocks())):
        X,Y = sbp.grid.get_block(i)
        force1.append(np.array([force1_func(t,X,Y)]))
        force2.append(np.array([force2_func(t,X,Y)]))
        force3.append(np.array([force3_func(t,X,Y)]))

    return np.array([np.array([force1,force2,force3]).flatten(), 0], dtype=object)


def ins_operator(sbp, state, e = 0):
    """ The Euler spatial operator.
    Arguments:
        sbp   - A MultilbockSBP object.
        state - A state vector
        e     - diffusion coefficient

    Returns:
        S - The euler operator evaluated at the given state.
        J - The Jacobian of S at the given state.
        ux,uy,vx,vy - gradiens to be used in bbc-operators
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

    if e!= 0:
        uxx = sbp.diffx(dudx)
        uyy = sbp.diffy(dudy)

        vxx = sbp.diffx(dvdx)
        vyy = sbp.diffy(dvdy)

        l1 -= e*(uxx + uyy)
        l2 -= e*(uxx + uyy)

    L = np.array([l1,l2,l3]).flatten()

    grad_u = np.array([dudx,dudy]).flatten()
    grad_v = np.array([dvdx,dvdy]).flatten()

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

    if e != 0:
        lap = e*(Dx*Dx + Dy*Dy)
        dl1du -= lap
        dl2dv -= lap

    J = sparse.bmat([[dl1du, dl1dv, dl1dp],
                     [dl2du, dl2dv, dl2dp],
                     [dl3du, dl3dv, dl3dp]])

    J = sparse.csr_matrix(J)

    return np.array([L, J, dudx, dudy, dvdx, dvdy], dtype=object)


def wall_operator(sbp, state, block_idx, side, e = 0):
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

    # viscous SATs

    pinv_mat = sbp.get_full_pinv(block_idx)

    bd_quad_mat = sbp.grid.bd_func_to_grid_func(bd_quad, block_idx, side)
    bd_quad_mat = grid_function_to_diagonal_matrix(bd_quad_mat)

    lift_mat = sbp.grid.bd_func_to_grid_func(lift, block_idx, side)
    lift_mat = grid_function_to_diagonal_matrix(lift_mat)

    Nx_mat = sbp.grid.bd_func_to_grid_func(nx, block_idx, side)
    Nx_mat = grid_function_to_diagonal_matrix(Nx_mat)

    Ny_mat = sbp.grid.bd_func_to_grid_func(ny, block_idx, side)
    Ny_mat = grid_function_to_diagonal_matrix(Ny_mat)

    Dx  = sbp.get_Dx(block_idx)
    Dy  = sbp.get_Dy(block_idx)
    DnT = np.transpose(Nx_mat*Dx + Ny_mat*Dy)

    pen             = e*pinv_mat*DnT*bd_quad_mat
    S               = np.array([s1, s2, s3]).flatten()
    stop            = Nx*Ny
    S[0:stop]      += pen@u[block_idx].flatten()
    S[stop:2*stop] += pen@v[block_idx].flatten()

    #Jacobian
    ds1du = np.zeros((Nx,Ny))
    ds1du[bd_slice] = 0.5*lift*(u_bd*nx + wn)
    ds1du   = sparse.diags(ds1du.flatten()) - pen #<-- pen is contribution from INS
    ds1dv   = np.zeros((Nx,Ny))
    ds1dv[bd_slice] = 0.5*lift*u_bd*ny
    ds1dv = sparse.diags(ds1dv.flatten())
    ds1dp = np.zeros((Nx,Ny))
    ds1dp = sparse.diags(ds1dp.flatten())

    ds2du = np.zeros((Nx,Ny))
    ds2du[bd_slice] = 0.5*lift*v_bd*nx
    ds2du = sparse.diags(ds2du.flatten())
    ds2dv = np.zeros((Nx,Ny))
    ds2dv[bd_slice] = 0.5*lift*(v_bd*ny + wn) 
    ds2dv = sparse.diags(ds2dv.flatten()) - pen #<-- pen is contribution from INS
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


def inflow_operator(sbp, state, block_idx, side, wn_data, wt_data, e):
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

    pinv_mat = sbp.get_full_pinv(block_idx)
    
    bd_quad_mat = sbp.grid.bd_func_to_grid_func(bd_quad, block_idx, side)
    bd_quad_mat = grid_function_to_diagonal_matrix(bd_quad_mat)

    lift_mat = sbp.grid.bd_func_to_grid_func(lift, block_idx, side)
    lift_mat = grid_function_to_diagonal_matrix(lift_mat)

    Nx_mat = sbp.grid.bd_func_to_grid_func(nx, block_idx, side)
    Nx_mat = grid_function_to_diagonal_matrix(Nx_mat)

    Ny_mat = sbp.grid.bd_func_to_grid_func(ny, block_idx, side)
    Ny_mat = grid_function_to_diagonal_matrix(Ny_mat)

    wn_vec = sbp.grid.bd_func_to_grid_func(wn-wn_data, block_idx, side)\
             .toarray().flatten('F')

    wt_vec = sbp.grid.bd_func_to_grid_func(wt-wt_data, block_idx, side)\
             .toarray().flatten('F')

    Dx  = sbp.get_Dx(block_idx)
    Dy  = sbp.get_Dy(block_idx)
    DnT = np.transpose(Nx_mat*Dx + Ny_mat*Dy)

    pen             = e*pinv_mat*DnT*bd_quad_mat
    S               = np.array([s1, s2, s3]).flatten()
    stop            = Nx*Ny
    wn_mat          = Nx_mat@u[block_idx].flatten() + Ny_mat@v[block_idx].flatten()
    S[0:stop]      += pen*(Nx_mat@wn_vec - Ny_mat@wt_vec)
    S[stop:2*stop] += pen*(Ny_mat@wn_vec + Nx_mat@wt_vec)

    #Jacobian
    ds1du = np.zeros((Nx,Ny))
    ds1du[bd_slice] = -lift*(nx*nx*(2*wn-wn_data) + ny*ny*wn - nx*ny*(wt-wt_data))
    ds1du = sparse.diags(ds1du.flatten()) + pen*(Nx_mat*Nx_mat + Ny_mat*Ny_mat)
    ds1dv = np.zeros((Nx,Ny))
    ds1dv[bd_slice] = -lift*(nx*ny*(2*wn-wn_data) - ny*nx*wn - ny*ny*(wt-wt_data))
    ds1dv = sparse.diags(ds1dv.flatten()) + pen*(Nx_mat*Ny_mat - Ny_mat*Nx_mat)
    ds1dp = np.zeros((Nx,Ny))
    ds1dp = sparse.diags(ds1dp.flatten())

    ds2du = np.zeros((Nx,Ny))
    ds2du[bd_slice] = -lift*(nx*ny*(2*wn-wn_data) - nx*ny*wn + nx*nx*(wt-wt_data))
    ds2du = sparse.diags(ds2du.flatten()) + pen*(Ny_mat*Nx_mat - Nx_mat*Ny_mat)
    ds2dv = np.zeros((Nx,Ny))
    ds2dv[bd_slice] = -lift*(ny*ny*(2*wn-wn_data) + nx*nx*wn + nx*ny*(wt-wt_data))
    ds2dv = sparse.diags(ds2dv.flatten()) + pen*(Ny_mat*Ny_mat + Nx_mat*Nx_mat)
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


def pressure_operator(sbp, state, block_idx, side, ux, uy, vx, vy, \
                      normal_data = 0, tangential_data = 0, e = 0,):
    u,v,p = vec_to_tensor(sbp.grid, state)
    bd_slice = sbp.grid.get_boundary_slice(block_idx, side)
    normals = sbp.get_normals(block_idx, side)
    nx = normals[:,0]
    ny = normals[:,1]
    p_bd  = p[block_idx][bd_slice]
    u_bd  = u[block_idx][bd_slice]
    ux_bd = ux[block_idx][bd_slice]
    uy_bd = uy[block_idx][bd_slice]
    vx_bd = vx[block_idx][bd_slice]
    vy_bd = vy[block_idx][bd_slice]

    n_grad_u = nx*ux_bd + ny*uy_bd
    n_grad_v = nx*vx_bd + ny*vy_bd

    pinv    = sbp.get_pinv(block_idx, side)
    bd_quad = sbp.get_boundary_quadrature(block_idx, side)
    lift    = pinv*bd_quad
 
    Nx,Ny       = sbp.grid.get_shapes()[0]
    num_blocks  = sbp.grid.num_blocks

    s1 = np.zeros((num_blocks,Nx,Ny))
    s1[block_idx][bd_slice] = -lift*(nx*p_bd - e*n_grad_u - normal_data)
    s2 = np.zeros((num_blocks,Nx,Ny))
    s2[block_idx][bd_slice] = -lift*(ny*p_bd - e*n_grad_v - tangential_data)
    s3 = np.zeros((num_blocks,Nx,Ny))

    S = np.array([s1, s2, s3]).flatten()

    #Jacobian
    lift_mat = sbp.grid.bd_func_to_grid_func(lift, block_idx, side)
    lift_mat = grid_function_to_diagonal_matrix(lift_mat)

    Nx_mat = sbp.grid.bd_func_to_grid_func(nx, block_idx, side)
    Nx_mat = grid_function_to_diagonal_matrix(Nx_mat)

    Ny_mat = sbp.grid.bd_func_to_grid_func(ny, block_idx, side)
    Ny_mat = grid_function_to_diagonal_matrix(Ny_mat)

    Dx = sbp.get_Dx(block_idx)
    Dy = sbp.get_Dy(block_idx)

    Dn = Nx_mat*sbp.get_Dx(block_idx) + Ny_mat*sbp.get_Dy(block_idx)
    pen = e*lift_mat*Dn

    ds1du = -pen

    ds1dv = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds1dp = np.zeros((Nx,Ny))
    ds1dp[bd_slice] = lift*nx
    ds1dp = sparse.diags(ds1dp.flatten())

    ds2du = sparse.csr_matrix((Nx*Ny, Nx*Ny))

    ds2dv = -pen
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


def outflow_operator(sbp, state, block_idx, side, ux, uy, vx, vy, e = 0):
    ## for outflow boundary that might switch to inflow 

    u,v,p    = vec_to_tensor(sbp.grid, state)
    bd_slice = sbp.grid.get_boundary_slice(block_idx, side)
    normals = sbp.get_normals(block_idx, side)
    nx      = normals[:,0]
    ny      = normals[:,1]
    u_bd    = u[block_idx][bd_slice]
    v_bd    = v[block_idx][bd_slice]
    p_bd    = p[block_idx][bd_slice]
    ux_bd   = ux[block_idx][bd_slice]
    uy_bd   = uy[block_idx][bd_slice]
    vx_bd   = vx[block_idx][bd_slice]
    vy_bd   = vy[block_idx][bd_slice]

    n_grad_u = nx*ux_bd + ny*uy_bd
    n_grad_v = nx*vx_bd + ny*vy_bd

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
    s1[block_idx][bd_slice] = -lift*(nx*(h*w2+p_bd) - e*n_grad_u)
    s2[block_idx][bd_slice] = -lift*(ny*(h*w2+p_bd) - e*n_grad_v)

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

    lift_mat = sbp.grid.bd_func_to_grid_func(lift, block_idx, side)
    lift_mat = grid_function_to_diagonal_matrix(lift_mat)

    Nx_mat = sbp.grid.bd_func_to_grid_func(nx, block_idx, side)
    Nx_mat = grid_function_to_diagonal_matrix(Nx_mat)

    Ny_mat = sbp.grid.bd_func_to_grid_func(ny, block_idx, side)
    Ny_mat = grid_function_to_diagonal_matrix(Ny_mat)

    Dn = Nx_mat*sbp.get_Dx(block_idx) + Ny_mat*sbp.get_Dy(block_idx)
    pen = e*lift_mat*Dn

    ds1du += pen
    ds2dv += pen

    J = sparse.bmat([[ds1du, ds1dv, ds1dp],
                     [ds2du, ds2dv, ds2dp],
                     [ds3du, ds3dv, ds3dp]])

    return np.array([S,J], dtype=object)

def stabilized_natural_operator(sbp, state, block_idx, side, ux, uy, vx, vy, e = 0):
    ## for outflow boundary that might switch to inflow 

    u,v,p    = vec_to_tensor(sbp.grid, state)
    bd_slice = sbp.grid.get_boundary_slice(block_idx, side)
    normals = sbp.get_normals(block_idx, side)
    nx      = normals[:,0]
    ny      = normals[:,1]
    u_bd    = u[block_idx][bd_slice]
    v_bd    = v[block_idx][bd_slice]
    p_bd    = p[block_idx][bd_slice]
    ux_bd   = ux[block_idx][bd_slice]
    uy_bd   = uy[block_idx][bd_slice]
    vx_bd   = vx[block_idx][bd_slice]
    vy_bd   = vy[block_idx][bd_slice]

    n_grad_u = nx*ux_bd + ny*uy_bd
    n_grad_v = nx*vx_bd + ny*vy_bd

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
    s1[block_idx][bd_slice] = -lift*(h*u_bd + nx*p_bd - e*n_grad_u)
    s2[block_idx][bd_slice] = -lift*(h*v_bd + ny*p_bd - e*n_grad_v)

    S = np.array([s1, s2, s3]).flatten()

    #Jacobian
    dhdwn = -a*np.exp(a*wn+c)/((np.exp(a*wn) + np.exp(c))**2)
    dhdu  = dhdwn*nx
    dhdv  = dhdwn*ny

    #dw2du = 2*(wn*nx-wt*ny) # derivative of wn^2+wt^2 w.r.t. u
    #dw2dv = 2*(wn*ny+wt*nx) # derivative of wn^2+wt^2 w.r.t. v

    ds1du           = np.zeros((Nx,Ny))
    ds1du[bd_slice] = -lift*(dhdu*u_bd + h)
    ds1du           = sparse.diags(ds1du.flatten())

    ds1dv           = np.zeros((Nx,Ny))
    ds1dv[bd_slice] = -lift*dhdv*u_bd
    ds1dv           = sparse.diags(ds1dv.flatten())

    ds1dp           = np.zeros((Nx,Ny))
    ds1dp[bd_slice] = -lift*nx
    ds1dp           = sparse.diags(ds1dp.flatten())

    ds2du           = np.zeros((Nx,Ny))
    ds2du[bd_slice] = -lift*dhdu*v_bd
    ds2du           = sparse.diags(ds2du.flatten())

    ds2dv           = np.zeros((Nx,Ny))
    ds2dv[bd_slice] = -lift*(dhdv*v_bd + h)
    ds2dv           = sparse.diags(ds2dv.flatten())

    ds2dp           = np.zeros((Nx,Ny))
    ds2dp[bd_slice] = -lift*ny
    ds2dp           = sparse.diags(ds2dp.flatten())

    ds3du           = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds3dv           = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    ds3dp           = sparse.csr_matrix((Nx*Ny, Nx*Ny))

    lift_mat = sbp.grid.bd_func_to_grid_func(lift, block_idx, side)
    lift_mat = grid_function_to_diagonal_matrix(lift_mat)

    Nx_mat = sbp.grid.bd_func_to_grid_func(nx, block_idx, side)
    Nx_mat = grid_function_to_diagonal_matrix(Nx_mat)

    Ny_mat = sbp.grid.bd_func_to_grid_func(ny, block_idx, side)
    Ny_mat = grid_function_to_diagonal_matrix(Ny_mat)

    Dn = Nx_mat*sbp.get_Dx(block_idx) + Ny_mat*sbp.get_Dy(block_idx)
    pen = e*lift_mat*Dn

    ds1du += pen
    ds2dv += pen

    J = sparse.bmat([[ds1du, ds1dv, ds1dp],
                     [ds2du, ds2dv, ds2dp],
                     [ds3du, ds3dv, ds3dp]])

    return np.array([S,J], dtype=object)

def interior_low_operator(sbp, state, block_idx, lw_slice1, lw_slice2, uy, e):
    # impose LoW inside the computational domain
    u,v,p    = vec_to_tensor(sbp.grid, state)
    
    X,Y     = sbp.grid.get_block(0)
    y_lw    = Y[lw_slice1,lw_slice2]


    K       = y_lw*np.log(y_lw/mms.beta)
    sigma1  = 1
    sigma2  = np.zeros(u.shape)
    sigma2[block_idx][lw_slice1,lw_slice2] = -K*sigma1
    sigma2  = sigma2.flatten()

    s1  = np.zeros(u.shape)
    s1[block_idx][lw_slice1,lw_slice2] = u[block_idx][lw_slice1,lw_slice2] - \
                                         K*uy[block_idx][lw_slice1,lw_slice2]
    
    s1 = s1.flatten()

    Nx,Ny   = sbp.grid.get_shapes()[0]
    DyT = np.transpose(sbp.get_Dy(block_idx))
    Ei  = np.zeros((Nx,Ny))
    Ei[lw_slice1,lw_slice2]  = 1
    Ei  = sparse.diags(Ei.flatten())

    pen = sigma1*Ei + DyT*sparse.diags(sigma2)
    pen = sbp.get_full_pinv(block_idx)*pen
   
    s2  = np.zeros((Nx*Ny))
    s3  = np.zeros((Nx*Ny))
    S   = np.concatenate([pen*s1,s2,s3])
    
    #Jacobian
    ds1du  = np.zeros((Nx,Ny))
    ds1du[lw_slice1,lw_slice2]  = 1
    ds1du  = sparse.diags(ds1du.flatten())
    K_diag = np.zeros((Nx,Ny))
    K_diag[lw_slice1,lw_slice2]  = K
    K_diag  = sparse.diags(K_diag.flatten())
    ds1du  -= K_diag*sbp.get_Dy(block_idx)
    ds1du   = pen*ds1du
    
    ds = sparse.csr_matrix((Nx*Ny, Nx*Ny))

    J = sparse.bmat([[ds1du, ds,ds],
                     [ds,ds,ds],
                     [ds,ds,ds]])

    return np.array([-S,-J], dtype=object)


def backward_euler_step(op, prev_state, dt, tol):
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
    for k in range(10):
        L, J = F(new_state, prev_state)

        err = np.linalg.norm(L, ord=np.inf)
        if err < tol:
            break

        delta = sparse.linalg.spsolve(J,L)
        new_state -= delta
        n_iter += 1

    #print("Error: {:.2e}".format(err))
    return new_state


def implicit_integration(grid, spatial_operator, init_u, init_v, init_p, dt, num_steps):
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
    tol = 1e-11
    for k in tqdm(range(num_steps)):
        sol = backward_euler_step(spatial_operator, sol, dt, tol)

        sol = vec_to_tensor(grid, sol)
        U.append(sol[0][0])
        V.append(sol[1][0])
        P.append(sol[2][0])
        sol = sol.flatten()

    return U,V,P
