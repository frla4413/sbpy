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
from sbpy.solve import solve_ivp, solve_steady_state, backward_euler_step, sbp_in_time_step_dg
from sbpy.utils import export_to_tecplot

def flatten_multiblock_vector(vec):
    return np.concatenate([ u.flatten() for u in vec])


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

    Nx,Ny      = grid.get_shapes()[0]
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

    X,Y = sbp.grid.get_block(0)

    l1 = 0.5*(u*dudx + v*dudy + duudx + duvdy) + dpdx
    l2 = 0.5*(u*dvdx + v*dvdy + dvvdy + duvdx) + dpdy
    l3 = dudx + dvdy

    L = np.array([l1,l2,l3]).flatten()
        
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

def interface_operator(sbp, state, idx1, side1, idx2, side2, if_idx): 
    u,v,p = vec_to_tensor(sbp.grid, state)

    bd_slice1   = sbp.grid.get_boundary_slice(idx1, side1)
    normals1    = sbp.get_normals(idx1, side1)
    nx1         = normals1[:,0]
    ny1         = normals1[:,1]
    u_bd1       = u[idx1][bd_slice1]
    v_bd1       = v[idx1][bd_slice1]
    p_bd1       = p[idx1][bd_slice1]
    wn1         = u_bd1*nx1 + v_bd1*ny1
    wt1         = -u_bd1*ny1 + v_bd1*nx1
    slice1      = get_jacobian_slice(idx1, side1, sbp.grid, False)

    bd_slice2   = sbp.grid.get_boundary_slice(idx2, side2)
    normals2    = sbp.get_normals(idx2, side2)
    nx2         = normals2[:,0]
    ny2         = normals2[:,1]
    u_bd2       = u[idx2][bd_slice2]
    v_bd2       = v[idx2][bd_slice2]
    p_bd2       = p[idx2][bd_slice2]
    wn2         = u_bd2*nx2 + v_bd2*ny2
    wt2         = -u_bd2*ny2 + v_bd2*nx2

    Nx,Ny       = sbp.grid.get_shapes()[0]
    num_blocks  = sbp.grid.num_blocks
    s1          = np.zeros((num_blocks,Nx,Ny))
    s2          = np.zeros((num_blocks,Nx,Ny))
    s3          = np.zeros((num_blocks,Nx,Ny))
    slice2      = get_jacobian_slice(idx2, side2, sbp.grid, False)

    is_flipped = sbp.grid.is_flipped_interface(if_idx)
    if is_flipped:
        nx2         = np.flip(nx2)
        ny2         = np.flip(ny2)
        u_bd2       = np.flip(u_bd2)
        v_bd2       = np.flip(v_bd2)
        p_bd2       = np.flip(p_bd2)
        wn2         = np.flip(wn2)
        wt2         = np.flip(wt2)
        slice2      = get_jacobian_slice(idx2, side2, sbp.grid, True)

    ## idx1
    pinv        = sbp.get_pinv(idx1, side1)
    bd_quad     = sbp.get_boundary_quadrature(idx1, side1)
    lift        = pinv*bd_quad
    lift        = -0.5*lift

    alpha = 0.5

    ic1 = wn1 + wn2
    ic2 = wn1*(wt1 + wt2)
    ic3 = p_bd1 - p_bd2
    w1 = nx1*wn1 - alpha*(nx1*wn2 - ny1*wt2)
    w2 = ny1*wn1 - alpha*(ny1*wn2 + nx1*wt2)

    s1[idx1][bd_slice1] = lift*(w1*ic1 - ny1*ic2 + nx1*ic3)
    s2[idx1][bd_slice1] = lift*(w2*ic1 + nx1*ic2+ ny1*ic3)
    s3[idx1][bd_slice1] = lift*ic1

    sigma = 0 #REMEMBER SCALING IN LIFT!, sigma < 0 required
    ic22  = wt1 + wt2
    s1[idx1][bd_slice1] += sigma*lift*(nx1*ic1 - ny1*ic22)
    s2[idx1][bd_slice1] += sigma*lift*(ny1*ic1 + nx1*ic22)


    Nx,Ny = sbp.grid.get_shapes()[0]
    num_blocks = sbp.grid.num_blocks

    dw1du1 = nx1*nx1
    dw1du2 = -alpha*(nx1*nx2 + ny1*ny2)
    dw1dv1 = nx1*ny1
    dw1dv2 = alpha*(ny1*nx2 - nx1*ny2)

    dic1du1 = nx1
    dic1dv1 = ny1
    dic1du2 = nx2
    dic1dv2 = ny2

    dic2du1 = nx1*(wt1+wt2) - wn1*ny1
    dic2dv1 = ny1*(wt1+wt2) + wn1*nx1
    dic2du2 = -wn1*ny2
    dic2dv2 = wn1*nx2

    dw2du1  = ny1*nx1
    dw2dv1  = ny1*ny1
    dw2du2  = alpha*(nx1*ny2 - ny1*nx2)
    dw2dv2  = -alpha*(ny1*ny2 + nx1*nx2)

    ds1du1                  = np.zeros((num_blocks,Nx,Ny))
    ds1du1[idx1][bd_slice1] = lift*(dw1du1*ic1 + w1*dic1du1 - ny1*dic2du1 \
                                   + sigma*(nx1*nx1 + ny1*ny1))

    ds1du2                  = lift*(dw1du2*ic1 + w1*dic1du2 - ny1*dic2du2 \
                                    + sigma*(nx1*nx2 + ny1*ny2))

    ds1du                   = sparse.diags(ds1du1.flatten(), format='csr')
    ds1du[slice1,slice2]    = np.diag(ds1du2)

    ds1dv1                  = np.zeros((num_blocks,Nx,Ny))
    ds1dv1[idx1][bd_slice1] = lift*(dw1dv1*ic1 + w1*dic1dv1 - ny1*dic2dv1 \
                                    + sigma*(nx1*ny1 - ny1*nx1))
    
    ds1dv2                  = lift*(dw1dv2*ic1 + w1*dic1dv2 - ny1*dic2dv2 \
                                    + sigma*(nx1*ny2 - ny1*nx2))

    ds1dv                   = sparse.diags(ds1dv1.flatten(),format='csr')
    ds1dv[slice1,slice2]    = np.diag(ds1dv2)

    ds1dp1                  = np.zeros((num_blocks,Nx,Ny))
    ds1dp1[idx1][bd_slice1] = lift*nx1
    ds1dp2                  = -ds1dp1[idx1][bd_slice1]#0.5*lift1*nx1
    ds1dp                   = sparse.diags(ds1dp1.flatten(),format='csr')
    ds1dp[slice1,slice2]    = np.diag(ds1dp2)

    ds2du1                  = np.zeros((num_blocks,Nx,Ny))
    ds2du1[idx1][bd_slice1] = lift*(dw2du1*ic1 + w2*dic1du1 + nx1*dic2du1 \
                                    + sigma*(ny1*nx1 - nx1*ny1))

    ds2du2                  = lift*(dw2du2*ic1 + w2*dic1du2 + nx1*dic2du2 \
                                    + sigma*(ny1*nx2 - nx1*ny2))

    ds2du                   = sparse.diags(ds2du1.flatten(),format='csr')
    ds2du[slice1,slice2]    = np.diag(ds2du2)

    ds2dv1                  = np.zeros((num_blocks,Nx,Ny))
    ds2dv1[idx1][bd_slice1] = lift*(dw2dv1*ic1 + w2*dic1dv1 + nx1*dic2dv1 \
                                    + sigma*(ny1*ny1 + nx1*nx1))

    ds2dv2                  = lift*(dw2dv2*ic1 + w2*dic1dv2 + nx1*dic2dv2 \
                                    + sigma*(ny1*ny2 + nx1*nx2))

    ds2dv                   = sparse.diags(ds2dv1.flatten(),format='csr')
    ds2dv[slice1,slice2]    = np.diag(ds2dv2)

    ds2dp1                  = np.zeros((num_blocks,Nx,Ny))
    ds2dp1[idx1][bd_slice1] = lift*ny1
    ds2dp2                  = -ds2dp1[idx1][bd_slice1]
    ds2dp                   = sparse.diags(ds2dp1.flatten(),format='csr')
    ds2dp[slice1,slice2]    = np.diag(ds2dp2)

    ds3du1                  = np.zeros((num_blocks,Nx,Ny))
    ds3du1[idx1][bd_slice1] = lift*nx1
    ds3du2                  = lift*nx2
    ds3du                   = sparse.diags(ds3du1.flatten(),format='csr')
    ds3du[slice1,slice2]    = np.diag(ds3du2)
    
    ds3dv1                  = np.zeros((num_blocks,Nx,Ny))
    ds3dv1[idx1][bd_slice1] = lift*ny1
    ds3dv2                  = lift*ny2
    ds3dv                   = sparse.diags(ds3dv1.flatten(),format='csr')
    ds3dv[slice1,slice2]    = np.diag(ds3dv2)

    ds3dp = None

    if is_flipped:
        
        # flip back idx2
        nx2         = np.flip(nx2)
        ny2         = np.flip(ny2)
        u_bd2       = np.flip(u_bd2)
        v_bd2       = np.flip(v_bd2)
        p_bd2       = np.flip(p_bd2)
        wn2         = np.flip(wn2)
        wt2         = np.flip(wt2)
        slice2      = get_jacobian_slice(idx2, side2, sbp.grid, False)

        #flip idx1
        nx1         = np.flip(nx1)
        ny1         = np.flip(ny1)
        u_bd1       = np.flip(u_bd1)
        v_bd1       = np.flip(v_bd1)
        p_bd1       = np.flip(p_bd1)
        wn1         = np.flip(wn1)
        wt1         = np.flip(wt1)
        slice1      = get_jacobian_slice(idx1, side1, sbp.grid, True)

    ############ idx2 - coupling ####################
    pinv        = sbp.get_pinv(idx2, side2)
    bd_quad     = sbp.get_boundary_quadrature(idx2, side2)
    lift        = pinv*bd_quad
    lift        = -0.5*lift

    beta = 1 - alpha
    ic1  = wn1 + wn2
    ic2  = wn2*(wt1 + wt2)
    ic3  = p_bd2 - p_bd1
    w1   = nx2*wn2 - beta*(nx2*wn1 - ny2*wt1)
    w2   = ny2*wn2 - beta*(ny2*wn1 + nx2*wt1)

    s1[idx2][bd_slice2] += lift*(w1*ic1 - ny2*ic2 + nx2*ic3)
    s2[idx2][bd_slice2] += lift*(w2*ic1 + nx2*ic2 + ny2*ic3)
    s3[idx2][bd_slice2] += lift*ic1

    ic22  = wt1 + wt2
    s1[idx2][bd_slice2] += sigma*lift*(nx2*ic1 - ny2*ic22)
    s2[idx2][bd_slice2] += sigma*lift*(ny2*ic1 + nx2*ic22)


    S = np.array([s1,s2,s3]).flatten()

    dw1du1 = -beta*(nx2*nx1 + ny2*ny1)
    dw1dv1 = -beta*(nx2*ny1 - ny2*nx1)
    dw1du2 = nx2*nx2
    dw1dv2 = nx2*ny2
    
    dic1du1 = nx1
    dic1dv1 = ny1
    dic1du2 = nx2
    dic1dv2 = ny2

    dic2du1 = -wn2*ny1
    dic2dv1 = wn2*nx1
    dic2du2 = nx2*(wt1+wt2) - wn2*ny2
    dic2dv2 = ny2*(wt1+wt2) + wn2*nx2

    dw2du1  = beta*(nx2*ny1- ny2*nx1)
    dw2dv1  = -beta*(ny2*ny1 + nx2*nx1)
    dw2du2  = ny2*nx2
    dw2dv2  = ny2*ny2

    ds1du2                  = np.zeros((num_blocks,Nx,Ny))
    ds1du2[idx2][bd_slice2] = lift*(dw1du2*ic1 + w1*dic1du2 - ny2*dic2du2 \
                                    + sigma*(nx2*nx2 + ny2*ny2))

    ds1du1                  = lift*(dw1du1*ic1 + w1*dic1du1 - ny2*dic2du1 \
                                    + sigma*(nx2*nx1 + ny2*ny1))

    ds1du                  += sparse.diags(ds1du2.flatten())
    ds1du[slice2,slice1]   += sparse.diags(ds1du1)

    ds1dv2                  = np.zeros((num_blocks,Nx,Ny))
    ds1dv2[idx2][bd_slice2] = lift*(dw1dv2*ic1 + w1*dic1dv2 - ny2*dic2dv2 \
                                    + sigma*(nx2*ny2 - ny2*nx2))

    ds1dv1                  = lift*(dw1dv1*ic1 + w1*dic1dv1 - ny2*dic2dv1 \
                                    + sigma*(nx2*ny1 - ny2*nx1))

    ds1dv                  += sparse.diags(ds1dv2.flatten())
    ds1dv[slice2,slice1]   += sparse.diags(ds1dv1)

    ds1dp2                  = np.zeros((num_blocks,Nx,Ny))
    ds1dp2[idx2][bd_slice2] = lift*nx2
    ds1dp1                  = -ds1dp2[idx2][bd_slice2]
    ds1dp                  += sparse.diags(ds1dp2.flatten())
    ds1dp[slice2,slice1]   += sparse.diags(ds1dp1)

    ds2du2                  = np.zeros((num_blocks,Nx,Ny))
    ds2du2[idx2][bd_slice2] = lift*(dw2du2*ic1 + w2*dic1du2 + nx2*dic2du2 \
                                    + sigma*(ny2*nx2 - nx2*ny2))

    ds2du1                  = lift*(dw2du1*ic1 + w2*dic1du1 + nx2*dic2du1 + 
                                   + sigma*(ny2*nx1 - nx2*ny1))

    ds2du                  += sparse.diags(ds2du2.flatten())
    ds2du[slice2,slice1]   += sparse.diags(ds2du1)

    ds2dv2                  = np.zeros((num_blocks,Nx,Ny))
    ds2dv2[idx2][bd_slice2] = lift*(dw2dv2*ic1 + w2*dic1dv2 + nx2*dic2dv2 \
                                    + sigma*(ny2*ny2 + nx2*nx2))

    ds2dv1                  = lift*(dw2dv1*ic1 + w2*dic1dv1 + nx2*dic2dv1 \
                                    + sigma*(ny2*ny1 + nx2*nx1))

    ds2dv                   += sparse.diags(ds2dv2.flatten())
    ds2dv[slice2,slice1]    += sparse.diags(ds2dv1)

    ds2dp2                  = np.zeros((num_blocks,Nx,Ny))
    ds2dp2[idx2][bd_slice2] = lift*ny2
    ds2dp1                  = -ds2dp2[idx2][bd_slice2]
    ds2dp                  += sparse.diags(ds2dp2.flatten())
    ds2dp[slice2,slice1]   += sparse.diags(ds2dp1)

    ds3du2                  = np.zeros((num_blocks,Nx,Ny))
    ds3du2[idx2][bd_slice2] = lift*nx2
    ds3du1                  = lift*nx1
    ds3du                  += sparse.diags(ds3du2.flatten())
    ds3du[slice2,slice1]   += sparse.diags(ds3du1)
    
    ds3dv2                  = np.zeros((num_blocks,Nx,Ny))
    ds3dv2[idx2][bd_slice2] = lift*ny2
    ds3dv1                  = lift*ny1
    ds3dv                  += sparse.diags(ds3dv2.flatten())
    ds3dv[slice2,slice1]   += sparse.diags(ds3dv1)

    J = sparse.bmat([[ds1du, ds1dv, ds1dp],
                     [ds2du, ds2dv, ds2dp],
                     [ds3du, ds3dv, ds3dp]])

    #pdb.set_trace()
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

    Nx,Ny      = sbp.grid.get_shapes()[0]
    num_blocks = sbp.grid.num_blocks
    s1         = np.zeros((num_blocks,Nx,Ny))
    s2         = np.zeros((num_blocks,Nx,Ny))
    s3         = np.zeros((num_blocks,Nx,Ny))

    s1[block_idx][bd_slice] = -0.5*lift*u_bd*wn
    s2[block_idx][bd_slice] = -0.5*lift*v_bd*wn
    s3[block_idx][bd_slice] = -lift*wn

    S = np.array([s1, s2, s3]).flatten()

    #Jacobian
    ds1du                      = np.zeros((num_blocks,Nx,Ny))
    ds1du[block_idx][bd_slice] = 0.5*lift*(u_bd*nx + wn)

    ds1du                       = sparse.diags(ds1du.flatten())
    ds1dv                       = np.zeros((num_blocks,Nx,Ny))
    ds1dv[block_idx][bd_slice]  = 0.5*lift*u_bd*ny
    ds1dv                       = sparse.diags(ds1dv.flatten())
    ds1dp                       = np.zeros((num_blocks,Nx,Ny))
    ds1dp                       = sparse.diags(ds1dp.flatten())

    ds2du                       = np.zeros((num_blocks,Nx,Ny))
    ds2du[block_idx][bd_slice]  = 0.5*lift*v_bd*nx
    ds2du                       = sparse.diags(ds2du.flatten())
    ds2dv                       = np.zeros((num_blocks,Nx,Ny))
    ds2dv[block_idx][bd_slice]  = 0.5*lift*(v_bd*ny + wn) 
    ds2dv                       = sparse.diags(ds2dv.flatten())
    ds2dp                       = np.zeros((num_blocks,Nx,Ny))
    ds2dp                       = sparse.diags(ds2dp.flatten())

    ds3du                       = np.zeros((num_blocks,Nx,Ny))
    ds3du[block_idx][bd_slice]  = lift*nx
    ds3du                       = sparse.diags(ds3du.flatten())
    ds3dv                       = np.zeros((num_blocks,Nx,Ny))
    ds3dv[block_idx][bd_slice]  = lift*ny
    ds3dv                       = sparse.diags(ds3dv.flatten())
    ds3dp                       = np.zeros((num_blocks,Nx,Ny))
    ds3dp                       = sparse.diags(ds3dp.flatten())
    
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

    Nx,Ny       = sbp.grid.get_shapes()[0]
    num_blocks  = sbp.grid.num_blocks

    wn_data     = wn_data_func(sbp,block_idx,side,t)
    wt_data     = wt_data_func(sbp,block_idx,side,t)

    s1                      = np.zeros((num_blocks,Nx,Ny))
    s2                      = np.zeros((num_blocks,Nx,Ny))
    s3                      = np.zeros((num_blocks,Nx,Ny))

    s1[block_idx][bd_slice] = -lift*nx*wn*(wn-wn_data) \
                              +lift*ny*wn*(wt-wt_data)
    s2[block_idx][bd_slice] = -lift*ny*wn*(wn-wn_data) \
                              -lift*nx*wn*(wt-wt_data)
    s3[block_idx][bd_slice] = -lift*(wn-wn_data)

    S = np.array([s1, s2, s3]).flatten()

    #Jacobian
    ds1du                       = np.zeros((num_blocks,Nx,Ny))
    ds1du[block_idx][bd_slice]  = -lift*(nx*nx*(2*wn-wn_data) + \
                                        ny*ny*wn - nx*ny*(wt-wt_data))
    ds1du                       = sparse.diags(ds1du.flatten())
                                  
    ds1dv                       = np.zeros((num_blocks,Nx,Ny))
    ds1dv[block_idx][bd_slice]  = -lift*(nx*ny*(2*wn-wn_data) - \
                                    ny*nx*wn - ny*ny*(wt-wt_data))
    ds1dv                       = sparse.diags(ds1dv.flatten())
    ds1dp                       = np.zeros((num_blocks,Nx,Ny))
    ds1dp                       = sparse.diags(ds1dp.flatten())

    ds2du                       = np.zeros((num_blocks,Nx,Ny))
    ds2du[block_idx][bd_slice]  = -lift*(nx*ny*(2*wn-wn_data) - \
                                    nx*ny*wn + nx*nx*(wt-wt_data))
    ds2du                       = sparse.diags(ds2du.flatten())
    ds2dv                       = np.zeros((num_blocks,Nx,Ny))
    ds2dv[block_idx][bd_slice]  = -lift*(ny*ny*(2*wn-wn_data) + \
                                    nx*nx*wn + nx*ny*(wt-wt_data))
    ds2dv                       = sparse.diags(ds2dv.flatten())
    ds2dp                       = np.zeros((num_blocks,Nx,Ny))
    ds2dp                       = sparse.diags(ds2dp.flatten())

    ds3du                       = np.zeros((num_blocks,Nx,Ny))
    ds3du[block_idx][bd_slice]  = -lift*nx
    ds3du                       = sparse.diags(ds3du.flatten())
    ds3dv                       = np.zeros((num_blocks,Nx,Ny))
    ds3dv[block_idx][bd_slice]  = -lift*ny
    ds3dv                       = sparse.diags(ds3dv.flatten())
    ds3dp                       = np.zeros((num_blocks,Nx,Ny))
    ds3dp                       = sparse.diags(ds3dp.flatten())

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


    Nx,Ny       = sbp.grid.get_shapes()[0]
    num_blocks  = sbp.grid.num_blocks
    s1                      = np.zeros((num_blocks,Nx,Ny))
    s1[block_idx][bd_slice] = -lift*(nx*p_bd -  nx*p_data)
    s2                      = np.zeros((num_blocks,Nx,Ny))
    s2[block_idx][bd_slice] = -lift*(ny*p_bd - ny*p_data)
    s3                      = np.zeros((num_blocks,Nx,Ny))

    S = np.array([s1, s2, s3]).flatten()

    #Jacobian
    n_o_p                       = num_blocks*Nx*Ny
    ds1du                       = sparse.csr_matrix((n_o_p,n_o_p))
    ds1dv                       = None
    ds1dp                       = np.zeros((num_blocks,Nx,Ny))
    ds1dp[block_idx][bd_slice]  = lift*nx
    ds1dp                       = sparse.diags(ds1dp.flatten())

    ds2du                       = None

    ds2dv                       = sparse.csr_matrix((n_o_p,n_o_p))
    ds2dp                       = np.zeros((num_blocks,Nx,Ny))
    ds2dp[block_idx][bd_slice]  = lift*ny
    ds2dp                       = sparse.diags(ds2dp.flatten())

    ds3du = sparse.csr_matrix((n_o_p, n_o_p))
    ds3dv = None
    ds3dp = None

    J = -sparse.bmat([[ds1du, ds1dv, ds1dp],
                      [ds2du, ds2dv, ds2dp],
                      [ds3du, ds3dv, ds3dp]])

    return np.array([S,J], dtype=object)


def outflow_operator(sbp, state, block_idx, side):

    ## for outflow boundary that might switch to inflow 

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
    s1          = np.zeros((num_blocks,Nx,Ny))
    s2          = np.zeros((num_blocks,Nx,Ny))
    s3          = np.zeros((num_blocks,Nx,Ny))

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

    ds1du                       = np.zeros((num_blocks,Nx,Ny))
    ds1du[block_idx][bd_slice]  = -lift*nx*(dhdu*w2+h*dw2du)
    ds1du                       = sparse.diags(ds1du.flatten())

    ds1dv                       = np.zeros((num_blocks,Nx,Ny))
    ds1dv[block_idx][bd_slice]  = -lift*nx*(dhdv*w2+h*dw2dv)
    ds1dv                       = sparse.diags(ds1dv.flatten())

    ds1dp                       = np.zeros((num_blocks,Nx,Ny))
    ds1dp[block_idx][bd_slice]  = -lift*nx
    ds1dp                       = sparse.diags(ds1dp.flatten())

    ds2du                       = np.zeros((num_blocks,Nx,Ny))
    ds2du[block_idx][bd_slice]  = -lift*ny*(dhdu*w2+h*dw2du)
    ds2du                       = sparse.diags(ds2du.flatten())

    ds2dv                       = np.zeros((num_blocks,Nx,Ny))
    ds2dv[block_idx][bd_slice]  = -lift*ny*(dhdv*w2+h*dw2dv)
    ds2dv                       = sparse.diags(ds2dv.flatten())

    ds2dp                       = np.zeros((num_blocks,Nx,Ny))
    ds2dp[block_idx][bd_slice]  = -lift*ny
    ds2dp                       = sparse.diags(ds2dp.flatten())

    n_o_p                       = num_blocks*Nx*Ny
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
            S_bd, J_bd = inflow_operator(sbp, state, block_idx, side, \
                                    bd_data["wn_data"], bd_data["wt_data"], t)

        elif bd_info["type"] == "pressure":
            S_bd, J_bd =   pressure_operator(sbp, state, block_idx, side, \
                                            bd_data["p_data"],t)

        elif bd_info["type"] == "outflow":
            S_bd, J_bd =   outflow_operator(sbp, state, block_idx, side)

        elif bd_info["type"] == "wall":
            S_bd, J_bd = wall_operator(sbp, state, block_idx, side)

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

def spatial_op(t,state, sbp, bd_indices, bd_data, force = None):
    S,J      = euler_operator(sbp, state)
    Sbd, Jbd = boundary_op(t,state, sbp, bd_indices, bd_data)
    Sif, Jif = sum_interface_op(state, sbp)

    if force:
        force = force_operator(sbp,force["force1"],force["force2"],force["force3"],t)[0]
        S     -= force
    
    return S + Sbd + Sif, J + Jbd + Jif

def solve_euler_ibvp(sbp, bd_indices, bd_data, init_u, init_v, init_p, \
                dt, num_timesteps, force = None, name_base = None): 
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
    init         = np.array([init_u, init_v, init_p]).flatten()
    spatial_func = lambda t,state: spatial_op(t, state, sbp, bd_indices, bd_data, force)

    file_name = name_base + str(0)
    export_to_tecplot(sbp.grid,init_u,init_v,init_p,file_name)

    sol = init
    tol = 1e-12
    for k in tqdm(range(num_timesteps)):
        L = spatial_func(k*dt,sol)[0]
        print("res = " +str(np.linalg.norm(L, ord=np.inf)))

        sol = backward_euler_step(k*dt,spatial_func,sol,dt,tol,incompressible = True)
        #sol = sbp_in_time_step_dg(k*dt,spatial_func,sol,dt,tol)
        U,V,P = vec_to_tensor(sbp.grid, sol)
        file_name = name_base + str(k+1)
        export_to_tecplot(sbp.grid,U,V,P,file_name)

    L = spatial_func(k*dt,sol)[0]
    print("res = " +str(np.linalg.norm(L, ord=np.inf)))

    return vec_to_tensor(sbp.grid, sol)
    #return ivp_solution_to_euler_variables(sbp.grid, sol_array)

def solve_euler_steady_state(sbp, bd_indices, bd_data, init_u, init_v, init_p, \
                             force = None): 
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
    init         = np.array([init_u, init_v, init_p]).flatten()
    spatial_func = lambda state: spatial_op(0, state, sbp, bd_indices, bd_data, force)
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
