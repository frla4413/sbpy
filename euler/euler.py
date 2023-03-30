"""
Differential and SAT operators for Euler. A state should be thought of as a
numpy array [U, V, P], where U, V, P are multiblock functions
(i.e. lists of grid functions). The operators here operate on
flattened versions of such a state. I.e. state = np.array([U, V, P]).flatten().

A complete spatial operator can be built by using the euler_operator function
together with boundary operators, for example, let sbp be a MultiblockSBP.
Then:

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
#import mms_abl as mms
import mms as mms
from sbpy.abl_utils import compute_low_error_robin
import matplotlib.pyplot as plt
import time

def vec_to_tensor(grid, vec):
    shapes = grid.get_shapes()
    component_length = np.sum([Nx*Ny for (Nx,Ny) in shapes])

    vec = np.reshape(vec, (3, component_length))

    start = 0
    U = []
    V = []
    P = []
    for Nx, Ny in shapes:
        U.append(np.reshape(vec[0][start:(start+Nx*Ny)], (Nx,Ny)))
        V.append(np.reshape(vec[1][start:(start+Nx*Ny)], (Nx,Ny)))
        P.append(np.reshape(vec[2][start:(start+Nx*Ny)], (Nx,Ny)))

    return np.array([U, V, P])

#def grid_function_to_diagonal_matrix(F):
#    F_mat = F.reshape((F.shape[0]*F.shape[1],1),order = 'F')
#    F_mat = F_mat.toarray()
#    return sparse.diags(F_mat[:,0])


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

def euler_operator(sbp, state, e = 0, turb_model = False, lm_func = None):
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

    if e!=0:
        mu_t = 0
        if turb_model:
        # if rans in abl project is used

            rans = np.sqrt(dudx*dudx + 0.5*(dudy + dvdx)*(dudy + dvdx)
                           + dvdy*dvdy)
            Y = sbp.grid.get_blocks()
            Y = Y[0][1]
            mu_t = (lm_func(Y))**2*rans

        l1 -= sbp.diffx((e+mu_t)*dudx)
        l1 -= sbp.diffy((e+mu_t)*dudy)

        l2 -= sbp.diffx((e+mu_t)*dvdx)
        l2 -= sbp.diffy((e+mu_t)*dvdy)

    L = np.array([l1,l2,l3]).flatten()

    # Jacobian
    Dx = sbp.get_Dx(0)
    Dy = sbp.get_Dy(0)
    U = sparse.diags(u[0].flatten())
    V = sparse.diags(v[0].flatten())
    Ux = sparse.diags(dudx.flatten())
    Uy = sparse.diags(dudy.flatten())
    Vx = sparse.diags(dvdx.flatten())
    Vy = sparse.diags(dvdy.flatten())
    e_mat = e
    if turb_model:
        e_mat = sparse.diags((e+mu_t).flatten())

    lap = e_mat*(Dx*Dx + Dy*Dy)

    dl1du = 0.5*(U@Dx + Ux + V@Dy + 2*Dx@U + Dy@V) - lap #pre_computed['laplace']
    dl1dv = 0.5*(Uy + Dy@U)
    dl1dp = Dx
    dl2du = 0.5*(Vx + Dx@V)
    dl2dv = 0.5*(U@Dx + V@Dy + Vy + Dx@U + 2*Dy@V) - lap #pre_computed['laplace']
    dl2dp = Dy
    dl3du = Dx
    dl3dv = Dy
    dl3dp = None

    J = sparse.bmat([[dl1du, dl1dv, dl1dp],
                     [dl2du, dl2dv, dl2dp],
                     [dl3du, dl3dv, dl3dp]])

    J = sparse.csr_matrix(J)
    return np.array([L, J, dudx, dudy, dvdx, dvdy], dtype=object)


def wall_operator(sbp, state, block_idx, side, e = 0, 
        turb_model = False, lm_func = None):
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
    # viscous SATs

    if e!=0:
        pinv_mat = sbp.get_full_pinv(block_idx)

        bd_quad_mat = np.zeros((Nx,Ny))
        bd_quad_mat[bd_slice] = bd_quad
        bd_quad_mat = sparse.diags(bd_quad_mat.flatten())
                
        Nx_mat = np.zeros((Nx,Ny))
        Nx_mat[bd_slice] = nx
        Nx_mat = sparse.diags(Nx_mat.flatten())

        Ny_mat = np.zeros((Nx,Ny))
        Ny_mat[bd_slice] = ny
        Ny_mat = sparse.diags(Ny_mat.flatten())

        Dx  = sbp.get_Dx(block_idx)
        Dy  = sbp.get_Dy(block_idx)
        DnT = np.transpose(Nx_mat*Dx + Ny_mat*Dy)
        
        e_mat = e
        if turb_model:
            dudx = sbp.diffx(u)
            dudy = sbp.diffy(u)
            dvdx = sbp.diffx(v)
            dvdy = sbp.diffy(v)

            rans = np.sqrt(dudx*dudx + 0.5*(dudy + dvdx)*(dudy + dvdx)
                      + dvdy*dvdy)
            Y = sbp.grid.get_blocks()
            Y = Y[0][1]
            mu_t = (lm_func(Y))**2*rans
            e_mat = sparse.diags((e+mu_t).flatten())

        pen             = e_mat*pinv_mat*DnT*bd_quad_mat
        stop            = Nx*Ny
        S[0:stop]      += pen@u[block_idx].flatten()
        S[stop:2*stop] += pen@v[block_idx].flatten()

    # Jacobian
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

    if e!=0:
        ds1du -= pen
        ds2dv -= pen

    J = -sparse.bmat([[ds1du, ds1dv, ds1dp],
                      [ds2du, ds2dv, ds2dp],
                      [ds3du, ds3dv, ds3dp]])

    return np.array([S,J], dtype=object)


def inflow_operator(sbp, state, block_idx, side, wn_data, wt_data, 
                    e = 0, turb_model = False, lm_func = None):
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

    if e!=0:
        pinv_mat = sbp.get_full_pinv(block_idx)
        
#        bd_quad_mat = sbp.grid.bd_func_to_grid_func(bd_quad, block_idx, side)
#        bd_quad_mat = grid_function_to_diagonal_matrix(bd_quad_mat)

#        lift_mat = sbp.grid.bd_func_to_grid_func(lift, block_idx, side)
#        lift_mat = grid_function_to_diagonal_matrix(lift_mat)

#        Nx_mat = sbp.grid.bd_func_to_grid_func(nx, block_idx, side)
#        Nx_mat = grid_function_to_diagonal_matrix(Nx_mat)
#
#        Ny_mat = sbp.grid.bd_func_to_grid_func(ny, block_idx, side)
#        Ny_mat = grid_function_to_diagonal_matrix(Ny_mat)

#        wn_vec = sbp.grid.bd_func_to_grid_func(wn-wn_data, block_idx, side)\
#                 .toarray().flatten('F')

#        wt_vec = sbp.grid.bd_func_to_grid_func(wt-wt_data, block_idx, side)\
#                 .toarray().flatten('F')

        bd_quad_mat = np.zeros((Nx,Ny))
        bd_quad_mat[bd_slice] = bd_quad
        bd_quad_mat = sparse.diags(bd_quad_mat.flatten())
                
        Nx_mat = np.zeros((Nx,Ny))
        Nx_mat[bd_slice] = nx
        Nx_mat = sparse.diags(Nx_mat.flatten())

        Ny_mat = np.zeros((Nx,Ny))
        Ny_mat[bd_slice] = ny
        Ny_mat = sparse.diags(Ny_mat.flatten())
        
        wn_vec = np.zeros((Nx,Ny))
        wn_vec[bd_slice] = wn-wn_data
        wn_vec = wn_vec.flatten()

        wt_vec = np.zeros((Nx,Ny))
        wt_vec[bd_slice] = wt-wt_data
        wt_vec = wt_vec.flatten()

        Dx  = sbp.get_Dx(block_idx)
        Dy  = sbp.get_Dy(block_idx)
        DnT = np.transpose(Nx_mat*Dx + Ny_mat*Dy)
        
        e_mat = e
        if turb_model:
            dudx = sbp.diffx(u)
            dudy = sbp.diffy(u)
            dvdx = sbp.diffx(v)
            dvdy = sbp.diffy(v)

            rans = np.sqrt(dudx*dudx + 0.5*(dudy + dvdx)*(dudy + dvdx)
                           + dvdy*dvdy)
            Y = sbp.grid.get_blocks()
            Y = Y[0][1]
            mu_t = (lm_func(Y))**2*rans
            e_mat = sparse.diags((e+mu_t).flatten())

        pen             = e_mat*pinv_mat*DnT*bd_quad_mat
        S               = np.array([s1, s2, s3]).flatten()
        stop            = Nx*Ny
        #wn_mat          = Nx_mat@u[block_idx].flatten() +\
        #                  Ny_mat@v[block_idx].flatten()
        S[0:stop]      += pen*(Nx_mat@wn_vec - Ny_mat@wt_vec)
        S[stop:2*stop] += pen*(Ny_mat@wn_vec + Nx_mat@wt_vec)

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

    if e!=0:
        ds1du += pen*(Nx_mat*Nx_mat + Ny_mat*Ny_mat)
        ds1dv += pen*(Nx_mat*Ny_mat - Ny_mat*Nx_mat)
        ds2du += pen*(Ny_mat*Nx_mat - Nx_mat*Ny_mat)
        ds2dv += pen*(Ny_mat*Ny_mat + Nx_mat*Nx_mat)


    J = sparse.bmat([[ds1du, ds1dv, ds1dp],
                     [ds2du, ds2dv, ds2dp],
                     [ds3du, ds3dv, ds3dp]])

    return np.array([S,J], dtype=object)


def pressure_operator(sbp, state, block_idx, side, ux, uy, vx, vy, \
                      normal_data = 0, tangential_data = 0, 
                      e = 0, turb_model = False, lm_func = None):

    u,v,p = vec_to_tensor(sbp.grid, state)
    bd_slice = sbp.grid.get_boundary_slice(block_idx, side)
    normals = sbp.get_normals(block_idx, side)
    nx = normals[:,0]
    ny = normals[:,1]
    p_bd  = p[block_idx][bd_slice]

    pinv    = sbp.get_pinv(block_idx, side)
    bd_quad = sbp.get_boundary_quadrature(block_idx, side)
    lift    = pinv*bd_quad

    Nx,Ny       = sbp.grid.get_shapes()[0]
    num_blocks  = sbp.grid.num_blocks
 
    s1 = np.zeros((num_blocks,Nx,Ny))
    s1[block_idx][bd_slice] = -lift*(nx*p_bd - normal_data)
    s2 = np.zeros((num_blocks,Nx,Ny))
    s2[block_idx][bd_slice] = -lift*(ny*p_bd - tangential_data)
    s3 = np.zeros((num_blocks,Nx,Ny))

    if e!= 0:
        ux_bd = ux[block_idx][bd_slice]
        uy_bd = uy[block_idx][bd_slice]
        vx_bd = vx[block_idx][bd_slice]
        vy_bd = vy[block_idx][bd_slice]

        n_grad_u = nx*ux_bd + ny*uy_bd
        n_grad_v = nx*vx_bd + ny*vy_bd

        e_mat = e
        if turb_model:
            Y = sbp.grid.get_blocks()
            Y = Y[0][1]
            rans = np.sqrt(ux_bd*ux_bd + 0.5*(uy_bd + vx_bd)*(uy_bd + vx_bd)
                           + vy_bd*vy_bd)
            mu_t = (lm_func(Y)[0])**2*rans
            e_mat = sparse.diags((e+mu_t).flatten())

        s1[block_idx][bd_slice] += -lift*(- e_mat*n_grad_u)
        s2[block_idx][bd_slice] += -lift*(- e_mat*n_grad_v)

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

    if e!= 0:
        lift_mat = np.zeros((Nx,Ny))
        lift_mat[bd_slice] = lift
        lift_mat = sparse.diags(lift_mat.flatten())
                
        Nx_mat = np.zeros((Nx,Ny))
        Nx_mat[bd_slice] = nx
        Nx_mat = sparse.diags(Nx_mat.flatten())

        Ny_mat = np.zeros((Nx,Ny))
        Ny_mat[bd_slice] = ny
        Ny_mat = sparse.diags(Ny_mat.flatten())

        Dn = Nx_mat*sbp.get_Dx(block_idx) + Ny_mat*sbp.get_Dy(block_idx)

        e_mat = e
        if turb_model:
            rans = np.sqrt(ux*ux + 0.5*(uy + vx)*(uy + vx) + vy*vy)
            mu_t = (lm_func(Y))**2*rans
            e_mat = sparse.diags((e+mu_t).flatten())

        pen = e_mat*lift_mat*Dn
        ds1du = -pen
        ds2dv = -pen

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


def neumann_bc_operator(sbp, state, block_idx, side, e, data, 
                        turb_model = False, lm_func = None):

    # neumann bc for u
    # only to be used on squares at lower boundary

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
    # viscous SATs

    pinv_mat = sbp.get_full_pinv(block_idx)

    bd_quad_mat = np.zeros((Nx,Ny))
    bd_quad_mat[bd_slice] = bd_quad
    bd_quad_mat = sparse.diags(bd_quad_mat.flatten())
            
    Nx_mat = np.zeros((Nx,Ny))
    Nx_mat[bd_slice] = nx
    Nx_mat = sparse.diags(Nx_mat.flatten())

    Ny_mat = np.zeros((Nx,Ny))
    Ny_mat[bd_slice] = ny
    Ny_mat = sparse.diags(Ny_mat.flatten())

    Dx  = sbp.get_Dx(block_idx)
    Dy  = sbp.get_Dy(block_idx)
    DnT = np.transpose(Nx_mat*Dx + Ny_mat*Dy)
    
    e_mat = e
    if turb_model:
        blocks = sbp.grid.get_blocks()
        Y = blocks[0][1]
        ux = sbp.diffx([u])
        vx = sbp.diffx([v])
        uy = sbp.diffy([u])
        vy = sbp.diffy([v])
        rans = np.sqrt(ux*ux + 0.5*(uy + vx)*(uy + vx) + vy*vy)
        mu_t = (lm_func(Y))**2*rans
        e_mat = sparse.diags((e+mu_t).flatten())

    data_vec = np.zeros((Nx,Ny))
    data_vec[bd_slice] = data
    data_vec = data_vec.flatten()

    pen             = e_mat*pinv_mat*DnT*bd_quad_mat
    stop            = Nx*Ny
    pen_neumann     = e_mat*pinv_mat*bd_quad_mat
    S[0:stop]      += pen_neumann@(Dy@u[block_idx].flatten() - data_vec)
    S[stop:2*stop] += pen@v[block_idx].flatten()

    # Jacobian
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

    ds1du -= pen_neumann*DnT
    ds2dv -= pen

    J = -sparse.bmat([[ds1du, ds1dv, ds1dp],
                      [ds2du, ds2dv, ds2dp],
                      [ds3du, ds3dv, ds3dp]])













def wall_model_as_bc_operator(sbp, state, block_idx, side, e, K, Kinv, 
                              turb_model = False, lm_func = None):

    # should only be used with e .ne. 0

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
    # viscous SATs

    pinv_mat = sbp.get_full_pinv(block_idx)

    bd_quad_mat = np.zeros((Nx,Ny))
    bd_quad_mat[bd_slice] = bd_quad
    bd_quad_mat = sparse.diags(bd_quad_mat.flatten())
            
    Nx_mat = np.zeros((Nx,Ny))
    Nx_mat[bd_slice] = nx
    Nx_mat = sparse.diags(Nx_mat.flatten())

    Ny_mat = np.zeros((Nx,Ny))
    Ny_mat[bd_slice] = ny
    Ny_mat = sparse.diags(Ny_mat.flatten())

    Dx  = sbp.get_Dx(block_idx)
    Dy  = sbp.get_Dy(block_idx)
    DnT = np.transpose(Nx_mat*Dx + Ny_mat*Dy)
    
    e_mat = e
    if turb_model:
        blocks = sbp.grid.get_blocks()
        Y = blocks[0][1]
        ux = sbp.diffx([u])
        vx = sbp.diffx([v])
        uy = sbp.diffy([u])
        vy = sbp.diffy([v])
        rans = np.sqrt(ux*ux + 0.5*(uy + vx)*(uy + vx) + vy*vy)
        mu_t = (lm_func(Y))**2*rans
        e_mat = sparse.diags((e+mu_t).flatten())

    pen             = e_mat*pinv_mat*DnT*bd_quad_mat
    stop            = Nx*Ny
    pen_low         = -e_mat*pinv_mat*bd_quad_mat*Kinv*(scipy.sparse.eye(Nx*Ny) - K*Dy)
    #pen_low         = -e_mat*pinv_mat*DnT*bd_quad_mat*(scipy.sparse.eye(Nx*Ny) - K*Dy)
    S[0:stop]      += pen_low@u[block_idx].flatten()
    S[stop:2*stop] += pen@v[block_idx].flatten()

    # Jacobian
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

    ds1du -= pen_low
    ds2dv -= pen

    J = -sparse.bmat([[ds1du, ds1dv, ds1dp],
                      [ds2du, ds2dv, ds2dp],
                      [ds3du, ds3dv, ds3dp]])

    return np.array([S,J], dtype=object)


def interior_penalty_dirichlet_operator(sbp, state, block_idx, lw_slice1, lw_slice2, data):
    # impose LoW inside the computational domain

    u    = vec_to_tensor(sbp.grid, state)[0]
    #sigma= np.array([-100, -100])
    #sigma= np.array([-100, -100])
    sigma = -100

    s1   = np.zeros(u.shape)
    Nx,Ny = sbp.grid.get_shapes()[0]
    pinv = 1/np.amax([Nx,Ny])
    pen = sigma*pinv
    s1[block_idx][lw_slice1,lw_slice2] = pen*(u[block_idx][lw_slice1,lw_slice2] - \
                                                data[lw_slice1,lw_slice2])
    s1 = s1.flatten()

    Nx,Ny = sbp.grid.get_shapes()[0]
    s2  = np.zeros((Nx*Ny))
    s3  = np.zeros((Nx*Ny))
    S   = np.concatenate([s1,s2,s3])
    
    #Jacobian
    ds1du  = np.zeros((Nx,Ny))
    ds1du[lw_slice1,lw_slice2]  = pen
    ds1du  = sparse.diags(ds1du.flatten())
    ds = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    J = sparse.bmat([[ds1du, ds,ds],
                     [ds,ds,ds],
                     [ds,ds,ds]])

    return np.array([-S,-J], dtype=object)

def interior_penalty_neumann_operator(sbp, state, block_idx, lw_slice1, lw_slice2, data):
    # impose LoW inside the computational domain

    u    = vec_to_tensor(sbp.grid, state)[0]
    sigma= -0.01

    s1   = np.zeros(u.shape)
    Nx,Ny = sbp.grid.get_shapes()[0]
    #Pinv = 1/np.amax([Nx,Ny])
    Pinv = sbp.get_full_pinv(block_idx)
    Ei = np.zeros(u.shape[1:3])
    Ei[lw_slice1, lw_slice2] = 1
    Ei = sparse.diags(Ei.flatten())
    uy = sbp.diffy([u])
    s1[block_idx][lw_slice1,lw_slice2] = (uy[block_idx][lw_slice1,lw_slice2] -
                                          data[lw_slice1,lw_slice2])

    DyT = np.transpose(sbp.get_Dy(block_idx))
    Dy = sbp.get_Dy(block_idx)
    #s1 = (pen*DyT)@(s1.flatten())
    s1 = sigma*(DyT*Pinv)@(s1.flatten())

    Nx,Ny = sbp.grid.get_shapes()[0]
    s2 = np.zeros((Nx*Ny))
    s3 = np.zeros((Nx*Ny))
    S  = np.concatenate([s1, s2, s3])
    
    #Jacobian
    ds1du = sigma*Pinv*DyT*Ei*Dy
    ds = sparse.csr_matrix((Nx*Ny, Nx*Ny))
    J = sparse.bmat([[ds1du, ds,ds],
                     [ds,ds,ds],
                     [ds,ds,ds]])
    return np.array([-S,-J], dtype=object)


def interior_penalty_robin_operator(sbp, state, block_idx, lw_slice1, lw_slice2, uy, e, beta):
    # impose LoW inside the computational domain
    # penalty: u - K u_y = 0
    u    = vec_to_tensor(sbp.grid, state)[0]
    
    Y    = sbp.grid.get_block(0)[1]
    y_lw = Y[lw_slice1,lw_slice2]

    K       = y_lw*np.log(y_lw/beta)
    sigma1  = -1
    sigma2  = np.zeros(u.shape)
    sigma2[block_idx][lw_slice1,lw_slice2] = -K*sigma1
    sigma2  = sigma2.flatten()

    s1  = np.zeros(u.shape)
    s1[block_idx][lw_slice1,lw_slice2] = u[block_idx][lw_slice1,lw_slice2] - \
                                         K*uy[block_idx][lw_slice1,lw_slice2]
    
    s1 = s1.flatten()
    Nx,Ny = sbp.grid.get_shapes()[0]
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


def backward_euler_step(op, prev_state, dt, tol, t):
    N = int(len(prev_state)/3)

    def F(new_state, prev_state):
        T = np.concatenate(
                [(new_state[0:int(2*N)] - prev_state[0:int(2*N)]),
                 np.zeros(N)])
        S,Js = op(new_state, t + dt)

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

        if err > 1e7 or n_iter > 50: #temp magic constant
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

def bdf2_step(op, prev_state, prev_prev_state, dt, tol, t):
    N = int(len(prev_state)/3)

    def F(new_state, prev_state, prev_prev_state, t_new_state):
        T = np.concatenate(
            [(new_state[0:int(2*N)] - 
             (4/3)*prev_state[0:int(2*N)] +
             (1/3)*prev_prev_state[0:int(2*N)]),\
                 np.zeros(N)])
        S,Js = op(new_state, t_new_state)

        #Jacobian
        I = sparse.identity(N)
        O = sparse.csr_matrix((N,N))
        Jt = sparse.bmat([[I, O, O],
                          [O, I, O],
                          [O, O, O]])

        return T+(2/3)*dt*S, Jt+(2/3)*dt*Js

    new_state = prev_state.copy()
    n_iter = 0
    t_new_state = t + dt
    while True:
        L, J = F(new_state, prev_state, prev_prev_state, t_new_state)
        err = np.linalg.norm(L, ord=np.inf)
        if err < tol:
            break

        if err > 1e7 or n_iter > 50: #temp magic constant
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


def solve_with_mpt(grid, spatial_operator, init_u, init_v, init_p, dt, num_steps, sbp, mpt):
    """ Solves an Euler problem and adds internal penalties at each time step.
        Not automatic, must edit the program. 
    """
    P=[]
    U=[]
    V=[]
    sol = np.array([init_u, init_v, init_p]).flatten()
    error = []
    time_to_solve = []
    tol = 1e-10

    index_set = set()
    #time_tot = 0
    for k in range(num_steps):
        t = (k+1)*dt
        #start1 = time.time()
        if mpt:

            index_set.clear() # <-- use this if fixed ammount of penalty terms should be used
            sol = vec_to_tensor(grid, sol)
            u = sol[0][0]
            #start2 = time.time()
            low_error = np.abs(u - mms.u(t-dt, grid.get_X(0),grid.get_Y(0)))
            #low_error = compute_low_error_robin(sbp, u, 1)
            
            # index vector for sorting array in descending order
            ind_array = np.flip(low_error.argsort(axis=None))[0:1]
            ind_max = np.unravel_index(ind_array, low_error.shape)
            for i in range(len(ind_max[0])):
                index_set.add((ind_max[0][i], ind_max[1][i]))
            slice1 = []
            slice2 = []

            for ind in index_set:
                slice1.append(ind[0])
                slice2.append(ind[1])

            sol = sol.flatten()
        #time_tot += time.time() - start1
        #print("time1: " + str(time.time() - start1))
        #print("time2: " + str(time.time() - start2))
        def func(x,t):
            L,J = spatial_operator(x,t)
            ## dirichlet
            if mpt:
                data = mms.u(t,grid.get_X(0), grid.get_Y(0))
                Li, Ji =  interior_penalty_dirichlet_operator(sbp,x, 0, slice1, slice2, data)
                return L + Li, J+ Ji
            else:
                return L,J

        start = time.time()
        sol = backward_euler_step(func, sol, dt, tol, t)
        stop = time.time()
        time_to_solve.append(stop - start)
        #if k == 0:
        #    prev_state = sol
        #    sol = backward_euler_step(func, prev_state, dt, tol, t)
        #else:
        #    prev_prev_state = prev_state
        #    prev_state = sol
        #    sol = bdf2_step(func, prev_state, prev_prev_state, dt, tol, t)
        sol = vec_to_tensor(grid, sol)
        U.append(sol[0][0])
        V.append(sol[1][0])
        P.append(sol[2][0])
        err_u = sol[0][0] - mms.u(t,grid.get_X(0), grid.get_Y(0))
        error.append(np.sqrt(sbp.integrate([err_u*err_u])))
        sol = sol.flatten()
    #plt.scatter(slice1,slice2)
    #plt.show()

    #err = np.linalg.norm(spatial_operator(sol)[0], ord=np.inf)
    #print("Err: " + str(err))
    #print("tot time: " + str(time_tot))
    return U,V,P, error, time_to_solve

def solve(grid, spatial_operator, init_u, init_v, init_p, dt, num_steps, sbp):
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
    tol = 1e-10

    for k in tqdm(range(num_steps)):
        t = (k+1)*dt
        sol = backward_euler_step(spatial_operator, sol, dt, tol, t)
        #sol = vec_to_tensor(grid, sol)
        #if k == 0:
        #    prev_state = sol
        #    sol = backward_euler_step(spatial_operator, prev_state, dt, tol, t)
        #else:
        #    prev_prev_state = prev_state
        #    prev_state = sol
        #    sol = bdf2_step(spatial_operator, prev_state, prev_prev_state, dt, tol, t)

        sol = vec_to_tensor(grid, sol)
        U.append(sol[0][0])
        V.append(sol[1][0])
        P.append(sol[2][0])
        sol = sol.flatten()

    return U,V,P


def sbp_in_time_step(op, cur_state, dt, tol, t):
    N = int(len(cur_state)/3)

    def F(prev_state, next_state):
        T = np.concatenate([(next_state[0:2*N] - 0.75*prev_state[0:2*N] - 0.25*cur_state[0:2*N])/dt,
                     np.zeros(N),
                     (next_state[0:2*N] - prev_state[0:2*N])/dt,
                     np.zeros(N)])


        S0,Js0 = op(prev_state, t)
        S1,Js1 = op(next_state, t + dt)
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

def solve_steady_state(grid, spatial_operator, init_u, init_v,init_p, alpha_0 = 1):
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
    sol = vec_to_tensor(grid, sol)
    U.append(sol[0][0])
    V.append(sol[1][0])
    P.append(sol[2][0])
    sol = sol.flatten()

    err = np.linalg.norm(spatial_operator(sol)[0],ord=np.inf) 
    max_iter = 1900
    inner_iter = 15
    #tol      = err/1e8
    tol      = 1e-9
    print("tol: " + str(tol))
    delta    = sol
    for k in range(max_iter):
        alpha = alpha_0
        if err < 1:
            alpha = 1
        
        F,J = spatial_operator(sol)
        delta = sparse.linalg.spsolve(J, F)
        for i in range(inner_iter):
            tmp = sol - alpha*delta
            err_tmp = np.linalg.norm(spatial_operator(tmp)[0],ord=np.inf) 
            if err_tmp < err:
                break
            alpha = alpha / 2

        err = err_tmp
        sol = tmp
        if err < tol:
            sol = vec_to_tensor(grid, sol)
            U.append(sol[0][0])
            V.append(sol[1][0])
            P.append(sol[2][0])
            sol = sol.flatten()
            break

        if k % 40 == 0:
            print("error: ", err, " alpha ", alpha, "tol", tol)

        if i == inner_iter-1:
            print("Error not decreasing!")
            F = spatial_operator(sol)[0]
            print("Residual: ", np.linalg.norm(F,ord=np.inf)) 
            pdb.set_trace()
            break;

        if err > 1e8 or k > max_iter:
            raise Exception("Newton diverging, try decreasing alpha.")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                delta = sparse.linalg.spsolve(J, F)
        except sparse.linalg.MatrixRankWarning:
            print("Singular")
            pdb.set_trace()
            # Small perturbation if singular Jacobian
            delta = np.random.normal(scale=1e-7, size=len(sol))
            sol += delta

        sol = vec_to_tensor(grid, sol)
        U.append(sol[0][0])
        V.append(sol[1][0])
        P.append(sol[2][0])
        sol = sol.flatten()

    F,J = spatial_operator(sol)
    print("Final residual: ", np.linalg.norm(F,ord=np.inf)) 

    print("Quit in ", k, " iterations")
    return U,V,P

def solve_steady_state_newton_krylov(grid, spatial_operator, init_u, init_v,init_p):
    """ Solves the steady-state version of an Euler problem by a combination of Newton-Krylov and Newton iterations
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
    max_iter = 10
    tol      = 1e-9
    func_f = lambda x: spatial_operator(x)[0]
    func_f_ls = lambda x: 0.5*np.sum((spatial_operator(x)[0])**2)

    def func_fp_ls(x):
        F,J = spatial_operator(x)
        return F@J

    num_iter = 0
    err = 10
    while err > tol and num_iter < max_iter:
        sol = scipy.optimize.newton_krylov(func_f,sol,verbose=True, maxiter=150, f_tol = 1e-7)
        #sol = scipy.optimize.root(func_f, sol) #--> sol is not only the solution vector. Check sol.x
#        sol = scipy.optimize.least_squares(func_f_ls, sol,verbose = 2, jac=func_fp_ls)
        sol = vec_to_tensor(grid, sol)
        U.append(sol[0][0])
        V.append(sol[1][0])
        P.append(sol[2][0])
        sol = sol.flatten()
        err = np.max(np.abs(func_f(sol)))
        num_iter+= 1
    err = np.max(np.abs(func_f(sol)))
    print("Final error: " + str(err))
    return U,V,P


def solve_steady_state_gradient_descent(grid, spatial_operator, init_u, init_v,init_p):
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
    U.append(sol[0][0])
    V.append(sol[1][0])
    P.append(sol[2][0])

    max_iter = 500
    tol      = 1e-12
    alpha    = 1e-5
    for k in range(max_iter):

        F,J = spatial_operator(sol)

        err = np.linalg.norm(F,ord=np.inf) 
        if err < tol:
            break

        if  k % 20 == 0:
            print("error: ", err)

        #sol-= sparse.linalg.spsolve(J,F)
        #sol-= alpha*sparse.linalg.spsolve(J,F)
        #sol = (1-alpha)*sol + alpha*(sol - alpha*sparse.linalg.spsolve(J,F))
        sol -= alpha*np.transpose(J)*F
        sol = vec_to_tensor(grid, sol)
        U.append(sol[0][0])
        V.append(sol[1][0])
        P.append(sol[2][0])
        sol = sol.flatten()

    print(np.linalg.norm(F,ord=np.inf)) 

    print("Converged in ", k, "iterations")
    err = []

    # print errors for convergence rate in Newton steps
    for k in range(len(U)):
        err.append(np.linalg.norm(U[-1]-U[k],ord=np.inf))

    print("Error in inf-norm, u^* is the last iterate")
    print(err)
    return U,V,P
