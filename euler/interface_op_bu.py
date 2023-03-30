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

