import numpy as np
import scipy
from scipy import sparse

from sbpy import operators
import mms_abl as mms
from euler import euler_operator,wall_operator,vec_to_tensor, vec_to_tensor, grid_function_to_diagonal_matrix


def euler_operator_jacobian(sbp,grid,u,v,ux,uy,vx,vy,state, e = 0):

    state_u,state_v,state_p = vec_to_tensor(sbp.grid, state)

    state_ux = sbp.diffx(state_u)
    state_uy = sbp.diffy(state_u)

    jw_1  = 0.5*(u*state_ux + ux*state_u + v*state_uy +  sbp.diffy(v*state_u)) +\
    					   sbp.diffx(u*state_u)
    jw_1 += 0.5*(uy*state_v + sbp.diffy(u*state_v))
    jw_1 += sbp.diffx(state_p)

    state_vx = sbp.diffx(state_v)
    state_vy = sbp.diffy(state_v)

    jw_2 = 0.5*(vx*state_u + sbp.diffx(v*state_u))
    jw_2 += 0.5*(u*state_vx + v*state_vy + vy*state_v + sbp.diffx(u*state_v)) \
        + sbp.diffy(v*state_v)
    jw_2 += sbp.diffy(state_p)
    jw_3 = state_ux + state_vy

    if e != 0:
        jw_1 -= e*(sbp.diffx(state_ux) + sbp.diffy(state_uy))
        jw_2 -= e*(sbp.diffx(state_vx) + sbp.diffy(state_vy))

    return np.array([jw_1, jw_2, jw_3]).flatten()


def wall_operator_jacobian(sbp, grid, side,u,v,ux,uy,vx,vy, state, e = 0):

    # viscous SATs
    block_idx = 0
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

    #Jacobian
    ds1du = np.zeros((Nx,Ny))
    ds1du[bd_slice] = 0.5*lift*(u_bd*nx + wn)
    ds1du   = sparse.diags(ds1du.flatten()) #- pen #<-- pen is contribution from INS
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
    ds2dv = sparse.diags(ds2dv.flatten()) #- pen #<-- pen is contribution from INS
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

    if e != 0:
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

    		ds1du -= pen
    		ds2dv -= pen


    Jw = -sparse.bmat([[ds1du, ds1dv, ds1dp],
                      [ds2du, ds2dv, ds2dp],
                      [ds3du, ds3dv, ds3dp]]) * state
    return Jw


def inflow_operator_jacobian(sbp, grid,side, u,v, ux, uy, vx, vy, wn_data, wt_data, e = 0):

    block_idx = 0
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
    wn_mat          = Nx_mat@u[block_idx].flatten() + Ny_mat@v[block_idx].flatten()

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


    Jw = sparse.bmat([[ds1du, ds1dv, ds1dp],
                     [ds2du, ds2dv, ds2dp],
                     [ds3du, ds3dv, ds3dp]])*state

    return Jw

def pressure_operator_jacobian(sbp,grid,side,u,v,p,ux,uy,vx,vy,
                      normal_data = 0, tangential_data = 0, e = 0,):

    block_idx = 0
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
    Nx,Ny = sbp.grid.get_shapes()[0]
    num_blocks = sbp.grid.num_blocks

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

    Jw = -sparse.bmat([[ds1du, ds1dv, ds1dp],
                      [ds2du, ds2dv, ds2dp],
                      [ds3du, ds3dv, ds3dp]])*state
    return Jw
