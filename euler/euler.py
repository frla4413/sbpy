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

The system can then be integrated using the bdf function, or the
sbp_in_time function.
"""

import pdb
import warnings

import numpy as np
import scipy
from scipy import sparse
from tqdm import tqdm
from sbpy import operators
from sbpy.solve import solve_ivp, solve_steady_state, bdf1_step, bdf2_step, sbp_in_time_step_dg
from sbpy.utils import export_to_tecplot
import matplotlib.pyplot as plt
from grid2d import flatten_multiblock_vector,allocate_gridfunction

#def flatten_multiblock_vector(vec):
#    return np.concatenate([ u.flatten() for u in vec])


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
        start += Nx*Ny

    return np.array([U, V, P])

def get_jacobian_slice(block_idx, side, grid, is_flipped):
    ''' Retrurns a slice-object. 
        Used in interface_operator in pairs to access 
        interface-parts of the jacobian.

        Ex: jacobian[slice1,slice2] --> access the part of the Jacobian 
                                        where interface2 affects interface1, 
                                        which is in the rows of interface1
                                        and columns of interface2.''' 

    Nx,Ny      = grid.get_shapes()[block_idx]
    N          = Nx*Ny
    num_blocks = grid.num_blocks

    if side == 'w':
        start_ind   = block_idx*N
        step        = 1
        stop_ind    = start_ind + Ny

    elif side == 'e':
        start_ind   = (block_idx + 1)*N - Ny
        step        = 1
        stop_ind    = (block_idx + 1)*N

    elif side == 's':
        start_ind   = block_idx*N
        step        = Ny
        stop_ind    = (block_idx + 1)*N - Ny + 1

    elif side == 'n':
        start_ind   = block_idx*N + Ny-1
        step        = Ny
        stop_ind    = (block_idx + 1)*N

    if is_flipped:
        if side == 'w' or side == 's':
            temp      = start_ind
            start_ind = stop_ind-1
            step      = -step
            if temp == 0:
                stop_ind = None
            else:
                stop_ind  = temp-1

        if side == 'e' or side == 'n':
            temp      = stop_ind
            stop_ind  = start_ind-1
            start_ind = temp-1
            step      = -step

    return slice(start_ind, stop_ind, step)

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

    L = np.array([flatten_multiblock_vector(l1),
                  flatten_multiblock_vector(l2),
                  flatten_multiblock_vector(l3)]).flatten()

    Dx = scipy.sparse.block_diag([sbp.get_Dx(i) for i in range(len(u))], "csr")
    Dy = scipy.sparse.block_diag([sbp.get_Dy(i) for i in range(len(u))], "csr")

    U  = sparse.diags(flatten_multiblock_vector(u))
    V  = sparse.diags(flatten_multiblock_vector(v))
    Ux = sparse.diags(flatten_multiblock_vector(dudx))
    Uy = sparse.diags(flatten_multiblock_vector(dudy))
    Vx = sparse.diags(flatten_multiblock_vector(dvdx))
    Vy = sparse.diags(flatten_multiblock_vector(dvdy))

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

def interface_operator(sbp, state, idx1, side1, idx2, side2, if_idx):
    u,v,p = vec_to_tensor(sbp.grid, state)

    bd_slice1   = sbp.grid.get_boundary_slice(idx1, side1)
    u_bd1       = u[idx1][bd_slice1]
    v_bd1       = v[idx1][bd_slice1]
    p_bd1       = p[idx1][bd_slice1]
    slice1      = get_jacobian_slice(idx1, side1, sbp.grid, False)

    bd_slice2   = sbp.grid.get_boundary_slice(idx2, side2)
    u_bd2       = u[idx2][bd_slice2]
    v_bd2       = v[idx2][bd_slice2]
    p_bd2       = p[idx2][bd_slice2]

    s1          = allocate_gridfunction(sbp.grid)
    s2          = allocate_gridfunction(sbp.grid)
    s3          = allocate_gridfunction(sbp.grid)
    slice2      = get_jacobian_slice(idx2, side2, sbp.grid, False)

    ## idx1
    pinv        = sbp.get_pinv(idx1, side1)
    bd_quad     = sbp.get_boundary_quadrature(idx1, side1)
    lift        = -pinv*bd_quad

    u_bar = 0.5*(u_bd1 + u_bd2)
    v_bar = 0.5*(v_bd1 + v_bd2)
    u_jump = u_bd1 - u_bd2
    v_jump = v_bd1 - v_bd2
    p_jump = p_bd1 - p_bd2

    uL = 0.75*u_bar + u_jump/8
    s1[idx1][bd_slice1] = lift*(uL*u_jump + p_jump/2)
    s2[idx1][bd_slice1] = lift*(0.5*u_bar*v_jump + v_bar*u_jump/4 +
                                u_jump*v_jump/8)
    s3[idx1][bd_slice1] = lift*u_jump/2

    duLdu1 = 0.75*0.5 + 1/8
    duLdu2 = 0.75*0.5 - 1/8
    du_jumpdu1 = 1
    du_jumpdu2 = -1

    ds1du1                  = allocate_gridfunction(sbp.grid)
    ds1du1[idx1][bd_slice1] = lift*(duLdu1*u_jump + uL*du_jumpdu1)
    ds1du2                  = lift*(duLdu2*u_jump + uL*du_jumpdu2)
    ds1du                   = sparse.diags(flatten_multiblock_vector(ds1du1), format='csr')
    ds1du[slice1,slice2]    = np.diag(ds1du2)

    ds1dv1                  = allocate_gridfunction(sbp.grid)
    ds1dv                   = sparse.diags(flatten_multiblock_vector(ds1dv1), format='csr')

    ds1dp1                  = allocate_gridfunction(sbp.grid)
    ds1dp1[idx1][bd_slice1] = 0.5*lift
    ds1dp2                  = -0.5*lift
    ds1dp                   = sparse.diags(flatten_multiblock_vector(ds1dp1),format='csr')
    ds1dp[slice1,slice2]    = np.diag(ds1dp2)

    ds2du1                  = allocate_gridfunction(sbp.grid)
    ds2du1[idx1][bd_slice1] = lift*(v_bar/4 + v_jump*(1/4 + 1/8))
    ds2du2                  = lift*(-v_bar/4 + v_jump*(1/4-1/8))
    ds2du                   = sparse.diags(flatten_multiblock_vector(ds2du1),format='csr')
    ds2du[slice1,slice2]    = np.diag(ds2du2)

    ds2dv1                  = allocate_gridfunction(sbp.grid)
    ds2dv1[idx1][bd_slice1] = lift*(u_jump/8 + (u_bar/2 + u_jump/8))
    ds2dv2                  = lift*(u_jump/8 - (u_bar/2 + u_jump/8))
    ds2dv                   = sparse.diags(flatten_multiblock_vector(ds2dv1),format='csr')
    ds2dv[slice1,slice2]    = np.diag(ds2dv2)

    ds2dp1                  = allocate_gridfunction(sbp.grid)
    ds2dp                   = sparse.diags(flatten_multiblock_vector(ds2dp1),format='csr')

    ds3du1                  = allocate_gridfunction(sbp.grid)
    ds3du1[idx1][bd_slice1] = lift/2
    ds3du2                  = -lift/2
    ds3du                   = sparse.diags(flatten_multiblock_vector(ds3du1),format='csr')
    ds3du[slice1,slice2]    = np.diag(ds3du2)

    ds3dv1                  = allocate_gridfunction(sbp.grid)
    ds3dv                   = sparse.diags(flatten_multiblock_vector(ds3dv1),format='csr')

    ds3dp = None

    ############# idx2 - coupling ####################
    pinv        = sbp.get_pinv(idx2, side2)
    bd_quad     = sbp.get_boundary_quadrature(idx2, side2)
    lift        = -pinv*bd_quad

    uR = (0.75*u_bar - u_jump/8)

    s1[idx2][bd_slice2] += lift*(uR*u_jump + p_jump/2)
    s2[idx2][bd_slice2] += lift*(0.5*u_bar*v_jump +
                                 v_bar*u_jump/4 -
                                 u_jump*v_jump/8)
    s3[idx2][bd_slice2] += lift*u_jump/2

    duRdu1 = 0.75*0.5 - 1/8
    duRdu2 = 0.75*0.5 + 1/8

    S = np.array([flatten_multiblock_vector(s1),
                  flatten_multiblock_vector(s2),
                  flatten_multiblock_vector(s3)]).flatten()

    ds1du2                  = allocate_gridfunction(sbp.grid)
    ds1du2[idx2][bd_slice2] = lift*(duRdu2*u_jump + uR*du_jumpdu2)
    ds1du1                  = lift*(duRdu1*u_jump + uR*du_jumpdu1)

    ds1du                  += sparse.diags(flatten_multiblock_vector(ds1du2))
    ds1du[slice2,slice1]   += sparse.diags(ds1du1)

    ds1dp2                  = allocate_gridfunction(sbp.grid)
    ds1dp2[idx2][bd_slice2] = -0.5*lift
    ds1dp1                  = 0.5*lift
    ds1dp                  += sparse.diags(flatten_multiblock_vector(ds1dp2))
    ds1dp[slice2,slice1]   += sparse.diags(ds1dp1)

    ds2du2                  = allocate_gridfunction(sbp.grid)
    ds2du2[idx2][bd_slice2] = lift*(v_jump/4 - v_bar/4 + v_jump/8)
    ds2du1                  = lift*(v_jump/4 + v_bar/4 - v_jump/8)
    ds2du                  += sparse.diags(flatten_multiblock_vector(ds2du2))

    ds2du[slice2,slice1]   += sparse.diags(ds2du1)
    ds2dv2                  = allocate_gridfunction(sbp.grid)
    ds2dv2[idx2][bd_slice2] = lift*(-0.5*u_bar + u_jump/8+u_jump/8)
    ds2dv1                  = lift*(0.5*u_bar + u_jump/8 - u_jump/8)
    ds2dv                   += sparse.diags(flatten_multiblock_vector(ds2dv2))
    ds2dv[slice2,slice1]    += sparse.diags(ds2dv1)

    ds3du2                  = allocate_gridfunction(sbp.grid)
    ds3du2[idx2][bd_slice2] = -lift/2
    ds3du1                  = lift/2
    ds3du                  += sparse.diags(flatten_multiblock_vector(ds3du2))
    ds3du[slice2,slice1]   += sparse.diags(ds3du1)

    J = sparse.bmat([[ds1du, ds1dv, ds1dp],
                     [ds2du, ds2dv, ds2dp],
                     [ds3du, ds3dv, ds3dp]])

    return np.array([S,J], dtype=object)


def wall_operator(sbp, state, block_idx, side):

    u,v,p    = vec_to_tensor(sbp.grid, state)
    bd_slice = sbp.grid.get_boundary_slice(block_idx, side)
    normals  = sbp.get_normals(block_idx, side)
    nx       = normals[:,0]
    ny       = normals[:,1]
    u_bd     = u[block_idx][bd_slice]
    v_bd     = v[block_idx][bd_slice]
    wn       = u_bd*nx + v_bd*ny
    pinv     = sbp.get_pinv(block_idx, side)
    bd_quad  = sbp.get_boundary_quadrature(block_idx, side)
    lift     = pinv*bd_quad

    num_blocks = sbp.grid.num_blocks
    s1         = allocate_gridfunction(sbp.grid)
    s2         = allocate_gridfunction(sbp.grid)
    s3         = allocate_gridfunction(sbp.grid)

    s1[block_idx][bd_slice] = -0.5*lift*u_bd*wn
    s2[block_idx][bd_slice] = -0.5*lift*v_bd*wn
    s3[block_idx][bd_slice] = -lift*wn

    S = np.array([flatten_multiblock_vector(s1),
                  flatten_multiblock_vector(s2),
                  flatten_multiblock_vector(s3)]).flatten()

    #Jacobian
    ds1du                      = allocate_gridfunction(sbp.grid)
    ds1du[block_idx][bd_slice] = 0.5*lift*(u_bd*nx + wn)

    ds1du                       = sparse.diags(flatten_multiblock_vector(ds1du))
    ds1dv                       = allocate_gridfunction(sbp.grid)
    ds1dv[block_idx][bd_slice]  = 0.5*lift*u_bd*ny
    ds1dv                       = sparse.diags(flatten_multiblock_vector(ds1dv))
    ds1dp                       = allocate_gridfunction(sbp.grid)
    ds1dp                       = sparse.diags(flatten_multiblock_vector(ds1dp))

    ds2du                       = allocate_gridfunction(sbp.grid)
    ds2du[block_idx][bd_slice]  = 0.5*lift*v_bd*nx
    ds2du                       = sparse.diags(flatten_multiblock_vector(ds2du))
    ds2dv                       = allocate_gridfunction(sbp.grid)
    ds2dv[block_idx][bd_slice]  = 0.5*lift*(v_bd*ny + wn) 
    ds2dv                       = sparse.diags(flatten_multiblock_vector(ds2dv))
    ds2dp                       = allocate_gridfunction(sbp.grid)
    ds2dp                       = sparse.diags(flatten_multiblock_vector(ds2dp))

    ds3du                       = allocate_gridfunction(sbp.grid)
    ds3du[block_idx][bd_slice]  = lift*nx
    ds3du                       = sparse.diags(flatten_multiblock_vector(ds3du))
    ds3dv                       = allocate_gridfunction(sbp.grid)
    ds3dv[block_idx][bd_slice]  = lift*ny
    ds3dv                       = sparse.diags(flatten_multiblock_vector(ds3dv))
    ds3dp                       = allocate_gridfunction(sbp.grid)
    ds3dp                       = sparse.diags(flatten_multiblock_vector(ds3dp))
    
    J = -sparse.bmat([[ds1du, ds1dv, ds1dp],
                      [ds2du, ds2dv, ds2dp],
                      [ds3du, ds3dv, ds3dp]])

    return np.array([S,J], dtype=object)

def jans_inflow_operator(sbp, state, block_idx, side, data):

    u,v,p    = vec_to_tensor(sbp.grid, state)
    bd_slice = sbp.grid.get_boundary_slice(block_idx, side)
    normals  = sbp.get_normals(block_idx, side)
    nx       = normals[:,0]
    ny       = normals[:,1]
    u_bd     = u[block_idx][bd_slice]
    v_bd     = v[block_idx][bd_slice]
    wn       = u_bd*nx + v_bd*ny
    pinv     = sbp.get_pinv(block_idx, side)
    bd_quad  = sbp.get_boundary_quadrature(block_idx, side)
    lift     = pinv*bd_quad

    num_blocks = sbp.grid.num_blocks
    s1         = allocate_gridfunction(sbp.grid)#allocate_gridfunction(sbp.grid)
    s2         = allocate_gridfunction(sbp.grid)#allocate_gridfunction(sbp.grid)
    s3         = allocate_gridfunction(sbp.grid)#allocate_gridfunction(sbp.grid)

    s1[block_idx][bd_slice] = -0.5*lift*u_bd*(wn - data)
    s2[block_idx][bd_slice] = -0.5*lift*v_bd*(wn - data)
    s3[block_idx][bd_slice] = -lift*(wn - data)

    S = np.array([flatten_multiblock_vector(s1),
                  flatten_multiblock_vector(s2), 
                  flatten_multiblock_vector(s3)]).flatten()

    #Jacobian
    ds1du                      = allocate_gridfunction(sbp.grid)
    ds1du[block_idx][bd_slice] = 0.5*lift*(u_bd*nx + (wn - data))

    ds1du                       = sparse.diags(flatten_multiblock_vector(ds1du))
    ds1dv                       = allocate_gridfunction(sbp.grid)
    ds1dv[block_idx][bd_slice]  = 0.5*lift*u_bd*ny
    ds1dv                       = sparse.diags(flatten_multiblock_vector(ds1dv))
    ds1dp                       = allocate_gridfunction(sbp.grid)
    ds1dp                       = sparse.diags(flatten_multiblock_vector(ds1dp))

    ds2du                       = allocate_gridfunction(sbp.grid)
    ds2du[block_idx][bd_slice]  = 0.5*lift*v_bd*nx
    ds2du                       = sparse.diags(flatten_multiblock_vector(ds2du))
    ds2dv                       = allocate_gridfunction(sbp.grid)
    ds2dv[block_idx][bd_slice]  = 0.5*lift*(v_bd*ny + (wn - data))
    ds2dv                       = sparse.diags(flatten_multiblock_vector(ds2dv))
    ds2dp                       = allocate_gridfunction(sbp.grid)
    ds2dp                       = sparse.diags(flatten_multiblock_vector(ds2dp))

    ds3du                       = allocate_gridfunction(sbp.grid)
    ds3du[block_idx][bd_slice]  = lift*nx
    ds3du                       = sparse.diags(flatten_multiblock_vector(ds3du))
    ds3dv                       = allocate_gridfunction(sbp.grid)
    ds3dv[block_idx][bd_slice]  = lift*ny
    ds3dv                       = sparse.diags(flatten_multiblock_vector(ds3dv))
    ds3dp                       = allocate_gridfunction(sbp.grid)
    ds3dp                       = sparse.diags(flatten_multiblock_vector(ds3dp))
    
    J = -sparse.bmat([[ds1du, ds1dv, ds1dp],
                      [ds2du, ds2dv, ds2dp],
                      [ds3du, ds3dv, ds3dp]])

    return np.array([S,J], dtype=object)



def inflow_operator(sbp, state, block_idx, side, wn_data_func, wt_data_func, t = 0):
    u,v,p       = vec_to_tensor(sbp.grid, state)
    bd_slice    = sbp.grid.get_boundary_slice(block_idx, side)
    normals     = sbp.get_normals(block_idx, side)
    nx          = normals[:,0]
    ny          = normals[:,1]
    u_bd        = u[block_idx][bd_slice]
    v_bd        = v[block_idx][bd_slice]
    wn          = u_bd*nx + v_bd*ny
    wt          = -u_bd*ny + v_bd*nx
    pinv        = sbp.get_pinv(block_idx, side)
    bd_quad     = sbp.get_boundary_quadrature(block_idx, side)
    lift        = pinv*bd_quad

    num_blocks  = sbp.grid.num_blocks

    wn_data     = wn_data_func(sbp,block_idx,side,t)
    wt_data     = wt_data_func(sbp,block_idx,side,t)
    s1         = allocate_gridfunction(sbp.grid)
    s2         = allocate_gridfunction(sbp.grid)
    s3         = allocate_gridfunction(sbp.grid)

    s1[block_idx][bd_slice] = -lift*nx*wn*(wn-wn_data) \
                              +lift*ny*wn*(wt-wt_data)
    s2[block_idx][bd_slice] = -lift*ny*wn*(wn-wn_data) \
                              -lift*nx*wn*(wt-wt_data)
    s3[block_idx][bd_slice] = -lift*(wn-wn_data)

    S = np.array([flatten_multiblock_vector(s1),
                  flatten_multiblock_vector(s2),
                  flatten_multiblock_vector(s3)]).flatten()
    #Jacobian
    ds1du                       = allocate_gridfunction(sbp.grid)
    ds1du[block_idx][bd_slice]  = -lift*(nx*nx*(2*wn-wn_data) + \
                                        ny*ny*wn - nx*ny*(wt-wt_data))
    ds1du                       = sparse.diags(flatten_multiblock_vector(ds1du))

    ds1dv                       = allocate_gridfunction(sbp.grid)
    ds1dv[block_idx][bd_slice]  = -lift*(nx*ny*(2*wn-wn_data) - \
                                    ny*nx*wn - ny*ny*(wt-wt_data))
    ds1dv                       = sparse.diags(flatten_multiblock_vector(ds1dv))
    ds1dp                       = allocate_gridfunction(sbp.grid)
    ds1dp                       = sparse.diags(flatten_multiblock_vector(ds1dp))

    ds2du                       = allocate_gridfunction(sbp.grid)
    ds2du[block_idx][bd_slice]  = -lift*(nx*ny*(2*wn-wn_data) - \
                                    nx*ny*wn + nx*nx*(wt-wt_data))
    ds2du                       = sparse.diags(flatten_multiblock_vector(ds2du))
    ds2dv                       = allocate_gridfunction(sbp.grid)
    ds2dv[block_idx][bd_slice]  = -lift*(ny*ny*(2*wn-wn_data) + \
                                    nx*nx*wn + nx*ny*(wt-wt_data))
    ds2dv                       = sparse.diags(flatten_multiblock_vector(ds2dv))
    ds2dp                       = allocate_gridfunction(sbp.grid)
    ds2dp                       = sparse.diags(flatten_multiblock_vector(ds2dp))

    ds3du                       = allocate_gridfunction(sbp.grid)
    ds3du[block_idx][bd_slice]  = -lift*nx
    ds3du                       = sparse.diags(flatten_multiblock_vector(ds3du))
    ds3dv                       = allocate_gridfunction(sbp.grid)
    ds3dv[block_idx][bd_slice]  = -lift*ny
    ds3dv                       = sparse.diags(flatten_multiblock_vector(ds3dv))
    ds3dp                       = allocate_gridfunction(sbp.grid)
    ds3dp                       = sparse.diags(flatten_multiblock_vector(ds3dp))

    J = sparse.bmat([[ds1du, ds1dv, ds1dp],
                     [ds2du, ds2dv, ds2dp],
                     [ds3du, ds3dv, ds3dp]])

    return np.array([S,J], dtype=object)

def pressure_operator(sbp, state, block_idx, side, p_data_func = 0, t = 0):
    u,v,p       = vec_to_tensor(sbp.grid, state)
    bd_slice    = sbp.grid.get_boundary_slice(block_idx, side)
    normals     = sbp.get_normals(block_idx, side)
    nx          = normals[:,0]
    ny          = normals[:,1]
    p_bd        = p[block_idx][bd_slice]
    u_bd        = u[block_idx][bd_slice]
    pinv        = sbp.get_pinv(block_idx, side)
    bd_quad     = sbp.get_boundary_quadrature(block_idx, side)
    lift        = pinv*bd_quad

    if p_data_func != 0:
        p_data      = p_data_func(sbp, block_idx, side, t)
    else: 
        p_data = 0


    num_blocks  = sbp.grid.num_blocks
    s1                      = allocate_gridfunction(sbp.grid)
    s1[block_idx][bd_slice] = -lift*(nx*p_bd -  nx*p_data)
    s2                      = allocate_gridfunction(sbp.grid)
    s2[block_idx][bd_slice] = -lift*(ny*p_bd - ny*p_data)
    s3                      = allocate_gridfunction(sbp.grid)

    S = np.array([flatten_multiblock_vector(s1),
                  flatten_multiblock_vector(s2),
                  flatten_multiblock_vector(s3)]).flatten()

    #Jacobian
    n_o_p                       = np.sum([Nx*Ny for (Nx,Ny) in sbp.grid.get_shapes()])
    ds1du                       = sparse.csr_matrix((n_o_p,n_o_p))
    ds1dv                       = None
    ds1dp                       = allocate_gridfunction(sbp.grid)
    ds1dp[block_idx][bd_slice]  = lift*nx
    ds1dp                       = sparse.diags(flatten_multiblock_vector(ds1dp))

    ds2du                       = None

    ds2dv                       = sparse.csr_matrix((n_o_p,n_o_p))
    ds2dp                       = allocate_gridfunction(sbp.grid)
    ds2dp[block_idx][bd_slice]  = lift*ny
    ds2dp                       = sparse.diags(flatten_multiblock_vector(ds2dp))

    ds3du = sparse.csr_matrix((n_o_p, n_o_p))
    ds3dv = None
    ds3dp = None

    J = -sparse.bmat([[ds1du, ds1dv, ds1dp],
                      [ds2du, ds2dv, ds2dp],
                      [ds3du, ds3dv, ds3dp]])

    return np.array([S,J], dtype=object)


def outflow_operator(sbp, state, block_idx, side):
    ## for outflow boundaries that might switch to inflow boundaries

    u,v,p       = vec_to_tensor(sbp.grid, state)
    bd_slice    = sbp.grid.get_boundary_slice(block_idx, side)
    normals     = sbp.get_normals(block_idx, side)
    nx          = normals[:,0]
    ny          = normals[:,1]
    u_bd        = u[block_idx][bd_slice]
    v_bd        = v[block_idx][bd_slice]
    p_bd        = p[block_idx][bd_slice]
    wn          = u_bd*nx + v_bd*ny
    wt          = -u_bd*ny + v_bd*nx
    pinv        = sbp.get_pinv(block_idx, side)
    bd_quad     = sbp.get_boundary_quadrature(block_idx, side)
    lift        = pinv*bd_quad

    Nx,Ny       = sbp.grid.get_shapes()[0]
    num_blocks  = sbp.grid.num_blocks
    s1          = allocate_gridfunction(sbp.grid)
    s2          = allocate_gridfunction(sbp.grid)
    s3          = allocate_gridfunction(sbp.grid)

    #parameters for sigmoid function, large a --> more steep curve, c --> shift sideways
    a = 50
    c = 0
    mag = 2
    h = mag*np.exp(c)/(np.exp(a*wn) + np.exp(c))

    w2 = wn*wn+wt*wt
    s1[block_idx][bd_slice] = -lift*nx*(h*w2+p_bd)
    s2[block_idx][bd_slice] = -lift*ny*(h*w2+p_bd)

    S = np.array([flatten_multiblock_vector(s1),
                  flatten_multiblock_vector(s2),
                  flatten_multiblock_vector(s3)]).flatten()


    #Jacobian
    dhdwn = -mag*a*np.exp(a*wn+c)/((np.exp(a*wn) + np.exp(c))**2)
    dhdu  = dhdwn*nx
    dhdv  = dhdwn*ny

    dw2du = 2*(wn*nx-wt*ny) # derivative of wn^2+wt^2 w.r.t. u
    dw2dv = 2*(wn*ny+wt*nx) # derivative of wn^2+wt^2 w.r.t. v

    ds1du                       = allocate_gridfunction(sbp.grid)
    ds1du[block_idx][bd_slice]  = -lift*nx*(dhdu*w2+h*dw2du)
    ds1du                       = sparse.diags(flatten_multiblock_vector(ds1du))

    ds1dv                       = allocate_gridfunction(sbp.grid)
    ds1dv[block_idx][bd_slice]  = -lift*nx*(dhdv*w2+h*dw2dv)
    ds1dv                       = sparse.diags(flatten_multiblock_vector(ds1dv))

    ds1dp                       = allocate_gridfunction(sbp.grid)
    ds1dp[block_idx][bd_slice]  = -lift*nx
    ds1dp                       = sparse.diags(flatten_multiblock_vector(ds1dp))

    ds2du                       = allocate_gridfunction(sbp.grid)
    ds2du[block_idx][bd_slice]  = -lift*ny*(dhdu*w2+h*dw2du)
    ds2du                       = sparse.diags(flatten_multiblock_vector(ds2du))

    ds2dv                       = allocate_gridfunction(sbp.grid)
    ds2dv[block_idx][bd_slice]  = -lift*ny*(dhdv*w2+h*dw2dv)
    ds2dv                       = sparse.diags(flatten_multiblock_vector(ds2dv))

    ds2dp                       = allocate_gridfunction(sbp.grid)
    ds2dp[block_idx][bd_slice]  = -lift*ny
    ds2dp                       = sparse.diags(flatten_multiblock_vector(ds2dp))

    n_o_p                       = np.sum([Nx*Ny for (Nx,Ny) in sbp.grid.get_shapes()])
    ds3du                       = sparse.csr_matrix((n_o_p, n_o_p))
    ds3dv                       = sparse.csr_matrix((n_o_p, n_o_p))
    ds3dp                       = sparse.csr_matrix((n_o_p, n_o_p))

    J = sparse.bmat([[ds1du, ds1dv, ds1dp],
                     [ds2du, ds2dv, ds2dp],
                     [ds3du, ds3dv, ds3dp]])

    return np.array([S,J], dtype=object)

def boundary_op(t, state, sbp, bd_indices, bd_data):
    N = len(state)
    Sbd = np.zeros(N)
    Jbd = sparse.csr_matrix((N, N))

    for (bd_idx, (block_idx, side)) in enumerate(sbp.grid.get_boundaries()):
        bd_info = sbp.grid.get_boundary_info(bd_idx)

        if bd_info["type"] == "inflow":
            S_bd, J_bd = inflow_operator(sbp, state, block_idx, side,
                                         bd_data["wn_data"], bd_data["wt_data"], t)

        elif bd_info["type"] == "pressure":
            S_bd, J_bd =   pressure_operator(sbp, state, block_idx, side,
                                             bd_data["p_data"],t)

        elif bd_info["type"] == "outflow":
            S_bd, J_bd =   outflow_operator(sbp, state, block_idx, side)

        elif bd_info["type"] == "wall":
            S_bd, J_bd = wall_operator(sbp, state, block_idx, side)

        elif bd_info["type"] == "jans_inflow_operator":
            S_bd, J_bd = jans_inflow_operator(sbp, state, block_idx, side,
                                              bd_data["jans_data"])

        elif bd_info["type"] == "periodic":
            S_bd = np.zeros(N)
            J_bd = sparse.csr_matrix((N, N))

        Sbd += S_bd
        Jbd += J_bd
    return Sbd, Jbd

def sum_interface_op(state, sbp):
    N = len(state)
    Sif = np.zeros(N)
    Jif = sparse.csr_matrix((N, N))

    for (if_idx,interface) in enumerate(sbp.grid.get_interfaces()):
        ((idx1,side1),(idx2,side2)) = interface

        S_if, J_if = interface_operator(sbp, state, idx1, side1, idx2, side2, if_idx)
        Sif += S_if
        Jif += J_if

    return Sif, Jif

def spatial_op(t,state, sbp, bd_indices, bd_data):
    S,J      = euler_operator(sbp, state)
    Sbd, Jbd = boundary_op(t,state, sbp, bd_indices, bd_data)
    Sif, Jif = sum_interface_op(state, sbp)

    return S + Sbd + Sif, J + Jbd + Jif

def solve_euler_ibvp(sbp, bd_indices, bd_data, init_u, init_v, init_p, \
                dt, num_timesteps, name_base = None): 
    """ Solve Euler equations

    Arguments:
        spatial_op: spatial operator also including the SATs
        grid:  MultiblockGridgrid-object
        sol_array: output from solve_ivp@solve.py
        init_u,v,p: multiblock function containing the initial data for u,v,p
        dt: time step
        num_timesteps: number of time steps

    Returns:
        U: A list of multiblock functions representing u in each time step.
        V: A list of multiblock functions representing v in each time step.
        P: A list of multiblock functions representing p in each time step.
    """
    init         = np.array([flatten_multiblock_vector(init_u),
                             flatten_multiblock_vector(init_v),
                             flatten_multiblock_vector(init_p)]).flatten()
    spatial_func = lambda t,state: spatial_op(t, state, sbp, bd_indices, bd_data)

    file_name = name_base + str(0)
    export_to_tecplot(sbp,init_u,init_v,init_p,file_name)

    sol = init
    tol = 1e-12
    sol_norm = []
    N = int(len(sol)/3)
    time_vec = []
    for k in tqdm(range(num_timesteps)):
        L = spatial_func(k*dt,sol)[0]

        if k == 0:
            prev_state = sol
            sol = bdf1_step(k*dt,spatial_func,prev_state,dt,tol,incompressible = True)
        else:
            prev_prev_state = prev_state
            prev_state = sol
            sol = bdf2_step(k*dt, spatial_func, prev_prev_state,
                            prev_state, dt, tol, incompressible = True)
        #sol = sbp_in_time_step_dg(k*dt,spatial_func,sol,dt,tol)
        U,V,P = vec_to_tensor(sbp.grid, sol)
        file_name = name_base + str(k+1)
        export_to_tecplot(sbp,U,V,P,file_name)
        sol_norm.append(scipy.linalg.norm(sol[:N],ord=np.inf))
        time_vec.append(k*dt)

#    L = spatial_func(k*dt,sol)[0]
#    plt.plot(time_vec,sol_norm)
#    plt.show()

    return vec_to_tensor(sbp.grid, sol)

def solve_euler_steady_state(sbp, bd_indices, bd_data, init_u, init_v, init_p): 
    """ Solve Euler equations

    Arguments:
        spatial_op: spatial operator also including the SATs
        grid:  MultiblockGridgrid-object
        sol_array: output from solve_ivp@solve.py
        init_u,v,p: multiblock function containing the initial guess for u,v,p

    Returns:
        U: A list of multiblock functions representing u in each time step.
        V: A list of multiblock functions representing v in each time step.
        P: A list of multiblock functions representing p in each time step.
    """
    init         = np.array([flatten_multiblock_vector(init_u),
                             flatten_multiblock_vector(init_v), 
                             flatten_multiblock_vector(init_p)]).flatten()
    spatial_func = lambda state: spatial_op(0, state, sbp, bd_indices, bd_data)
    sol_array    = solve_steady_state(spatial_func, init)

    return ivp_solution_to_euler_variables(sbp.grid, sol_array)


def ivp_solution_to_euler_variables(grid, sol_array):
    """ Convert sol_array from solve_ivp to U,V,P

    Arguments:
        grid:  MultiblockGridgrid-object
        sol_array: output from solve_ivp@solve.py

    Returns:
        U: A list of multiblock functions representing u in each time step.
        V: A list of multiblock functions representing v in each time step.
        P: A list of multiblock functions representing p in each time step.
    """
    P=[]
    U=[]
    V=[]
    for sol in sol_array:
        sol = vec_to_tensor(grid, sol)
        U.append(sol[0])
        V.append(sol[1])
        P.append(sol[2])
    return U,V,P
