import pdb
import numpy as np
from sbpy import dg_operators
from sbpy import grid2d

""" This module containes functions to generate meshes."""

def get_circle_sector_grid(N, th0, th1, r_inner, r_outer):
    """ Returns a circle sector grid.

    Arguments:
        N: Number of gridpoints in each direction.
        th0: Start angle.
        th1: End angle.
        r_inner: Inner radius.
        r_outer: Outer radius.

    Returns:
        (X,Y): A pair of matrices defining the grid.
    """
    d_r    = (r_outer - r_inner)/(N-1)
    d_th   = (th1-th0)/(N-1)
    radii  = np.linspace(r_inner, r_outer, N)
    thetas = np.linspace(th0, th1, N)

    x = np.zeros(N*N)
    y = np.zeros(N*N)

    pos = 0
    for r in radii:
        for th in thetas:
            x[pos] = r*np.cos(th)
            y[pos] = r*np.sin(th)
            pos += 1

    X = np.reshape(x,(N,N))
    Y = np.reshape(y,(N,N))

    return X,Y

def get_annulus_grid(N, r_in, r_out, num_blocks = 4):
    """ Returns a list of four blocks constituting an annulus grid. """
    blocks = []
    for i in range(num_blocks):
        blocks.append(get_circle_sector_grid(N, i*2*np.pi/num_blocks,\
                      (i+1)*2*np.pi/num_blocks,r_in,r_out))
    grid2d.collocate_corners(blocks)
    return blocks


def get_bump_grid(N):
    """ Returns a grid with two bumps in the floor and ceiling.
    Arguments:
        N: Number of gridpoints in each direction.

    Returns:
        (X,Y): A pair of matrices defining the grid.
    """
    x0 = -0.5
    x1 = 0.5
    dx = (x1-x0)/(N-1)
    y0 = lambda x: 0.0625*np.exp(-25*x**2)
    y1 = lambda x: 0.2 + 0.0625*np.exp(-25*x**2)
    x = np.zeros(N*N)
    y = np.copy(x)
    pos = 0
    for i in range(N):
        for j in range(N):
            x_val = x0 + i*dx
            x[pos] = x_val
            y[pos] = y0(x_val) + j*(y1(x_val)-y0(x_val))/(N-1)
            pos = pos+1

    X = np.reshape(x,(N,N))
    Y = np.reshape(y,(N,N))

    return X,Y

def get_channel_grid(N, x0 = -1.5, x1 = 1.5, y0_dat = 0, y1_dat = 0.8):
    """ Returns a grid with curved floor and ceiling.
    Arguments:
        N: Number of gridpoints in each direction.

    Returns:
        (X,Y): A pair of matrices defining the grid.
    """
    dx = (x1-x0)/(N-1)
    #y0 = lambda x: -1 + 0.0625*np.exp(-25*x**2 + 0.5)
    #y1 = lambda x: 1  - 0.0625*np.exp(-25*x**2 + 0.5)
    y0 = lambda x: y0_dat + 0.0625*np.exp(-25*x**2)
    y1 = lambda x: y1_dat + 0.0625*np.exp(-25*x**2)
    x = np.zeros(N*N)
    y = np.copy(x)
    pos = 0
    for i in range(N):
        for j in range(N):
            x_val = x0 + i*dx
            x[pos] = x_val
            y[pos] = y0(x_val) + j*(y1(x_val)-y0(x_val))/(N-1)
            pos = pos+1

    X = np.reshape(x,(N,N))
    Y = np.reshape(y,(N,N))

    return X,Y

""" 
    DG-grid below 
"""

def set_bd_info(grid, bd_info, boundary_condition):
    
    for (bd_idx, (block_idx, side)) in enumerate(grid.get_boundaries()):
        x_bd, y_bd = grid.get_boundary(block_idx,side)

        if contains_element(x_bd[0], bd_info["bd1_x"]) and \
           contains_element(y_bd[0], bd_info["bd1_y"]) and \
           contains_element(x_bd[-1], bd_info["bd1_x"]) and \
           contains_element(y_bd[-1], bd_info["bd1_y"]):
            grid.set_boundary_info(bd_idx, {'type': boundary_condition["bd1"]})

        elif contains_element(x_bd[0], bd_info["bd2_x"]) and \
             contains_element(y_bd[0], bd_info["bd2_y"]) and \
             contains_element(x_bd[-1], bd_info["bd2_x"]) and \
             contains_element(y_bd[-1],  bd_info["bd2_y"]):
            grid.set_boundary_info(bd_idx, {'type': boundary_condition["bd2"]})

        elif contains_element(x_bd[0], bd_info["bd3_x"]) and \
             contains_element(y_bd[0], bd_info["bd3_y"]) and \
             contains_element(x_bd[-1], bd_info["bd3_x"]) and \
             contains_element(y_bd[-1], bd_info["bd3_y"]):
            grid.set_boundary_info(bd_idx, {'type': boundary_condition["bd3"]})
        
        elif contains_element(x_bd[0], bd_info["bd4_x"]) and \
             contains_element(y_bd[0], bd_info["bd4_y"]) and \
             contains_element(x_bd[-1], bd_info["bd4_x"]) and \
             contains_element(y_bd[-1], bd_info["bd4_y"]):
            grid.set_boundary_info(bd_idx, {'type': boundary_condition["bd4"]})

    for (bd_idx, (block_idx, side)) in enumerate(grid.get_boundaries()):
        if grid.get_boundary_info(bd_idx) is None:
            print("bd_idx " + str(bd_idx) + " has no type!")

def contains_element(element, array): 
    out = False
    for i in array:
        if abs(element - i) < 1e-13:
            return True
    return False
            
def get_dg_grid(N,M,h,order):
    nodes = dg_operators.legendre_gauss_lobatto_nodes_and_weights(order)[0]
    [X,Y] = np.meshgrid(nodes,nodes)
    X     = 0.5*h*np.transpose(X) + 0.5*h  # X: from 0 -> h
    Y     = 0.5*h*np.transpose(Y) + 0.5*h  # Y: from 0 -> h
    blocks= []
    for i in range(N):
        for j in range(M):
            blocks.append((X + i*h,Y + j*h))

    grid2d.collocate_corners(blocks)
    return blocks

def get_dg_circle_sector_grid(N, th0, th1, r_inner, r_outer):
    d_r  = 0.5*(r_outer - r_inner)
    d_th = 0.5*(th1-th0)

    nodes = dg_operators.legendre_gauss_lobatto_nodes_and_weights(N-1)[0]
    radii = nodes*d_r + d_r + r_inner
    thetas= nodes*d_th + d_th + th0

    x = np.zeros(N*N)
    y = np.zeros(N*N)

    pos = 0
    for r in radii:
        for th in thetas:
            x[pos] = r*np.cos(th)
            y[pos] = r*np.sin(th)
            pos += 1

    X = np.reshape(x,(N,N))
    Y = np.reshape(y,(N,N))

    return X,Y

def get_annulus_dg_grid(N, r_in, r_out, num_blocks = 4):
    """ Returns a list of four blocks constituting an annulus grid with dg-points. """
    blocks = []
    for i in range(num_blocks):
        blocks.append(get_dg_circle_sector_grid(N, i*2*np.pi/num_blocks,\
                      (i+1)*2*np.pi/num_blocks,r_in,r_out))
    grid2d.collocate_corners(blocks)
    return blocks

def get_cylinder_grid(N, num_blocks):

    blocks      = []
    r_in        = 0.1
    dr          = 0.05
    r_out       = r_in + dr
    blocks_temp = get_annulus_dg_grid(N, r_in, r_out, num_blocks)
    for (X,Y) in blocks_temp:
        blocks.append([X,Y])


    for i in range(10):
        r_in        = r_out
        r_out       = r_in + dr
        blocks_temp = get_annulus_dg_grid(N, r_in, r_out, num_blocks)
        dr          = 2*dr
        for (X,Y) in blocks_temp:
            blocks.append([X,Y])

    print(r_out)
    grid2d.collocate_corners(blocks)
    return blocks

def get_cylinder_channel_grid(N, method):
    radius = 0.5
    limit  = 12
    M      = N

    nodes_xi  = dg_operators.legendre_gauss_lobatto_nodes_and_weights(N-1)[0]
    nodes_eta = dg_operators.legendre_gauss_lobatto_nodes_and_weights(M-1)[0]
    c1 = lambda xi:  np.array([(2 + j, 0) for j in xi])
    c2 = lambda eta: np.array([(3*(1-j)/2, 3*(1+j)/2) for j in eta])
    c3 = lambda xi:  np.array([(0, 2+j) for j in xi])
    c4 = lambda eta: np.array([(radius*np.cos(np.pi*(j + 1)/4), radius*np.sin(np.pi*(j + 1)/4)) for j in eta])

    curve  = c1(nodes_xi)
    G1     = dg_operators.CurveInterpolant(nodes_xi, curve[:,0], curve[:,1])
    curve  = c2(nodes_eta)
    G2     = dg_operators.CurveInterpolant(nodes_eta, curve[:,0], curve[:,1])
    curve  = c3(nodes_xi)
    G3     = dg_operators.CurveInterpolant(nodes_xi, curve[:,0], curve[:,1])
    curve  = c4(nodes_eta)
    G4     = dg_operators.CurveInterpolant(nodes_eta, curve[:,0], curve[:,1])
    
    X0,Y0  = transfinite_quad_map(G1, G2, G3, G4, method)
    X0     = X0.flatten()
    Y0     = Y0.flatten()

    blocks = []
    angle  = 5*np.pi/4
    for pos in range(len(X0)):
        x_temp = X0[pos]
        y_temp = Y0[pos]
        X0[pos] = np.cos(angle)*x_temp - np.sin(angle)*y_temp
        Y0[pos] = np.sin(angle)*x_temp + np.cos(angle)*y_temp
    blocks.append([np.reshape(X0,(N,M)),np.reshape(Y0,(N,M))])
 
    angle  = 0.5*np.pi
    X1 = np.zeros(N*M)
    Y1 = np.zeros(N*M)
    for pos in range(len(X0)):
        x_temp = X0[pos]
        y_temp = Y0[pos]
        X1[pos] = np.cos(angle)*x_temp - np.sin(angle)*y_temp
        Y1[pos] = np.sin(angle)*x_temp + np.cos(angle)*y_temp

    blocks.append([np.reshape(X1,(N,M)),np.reshape(Y1,(N,M))])
 
    X2 = np.zeros(N*M)
    Y2 = np.zeros(N*M)
    for pos in range(len(X0)):
        x_temp = X1[pos]
        y_temp = Y1[pos]
        X2[pos] = np.cos(angle)*x_temp - np.sin(angle)*y_temp
        Y2[pos] = np.sin(angle)*x_temp + np.cos(angle)*y_temp

    blocks.append([np.reshape(X2,(N,M)),np.reshape(Y2,(N,M))])

    X3 = np.zeros(N*M)
    Y3 = np.zeros(N*M)
    for pos in range(len(X0)):
        x_temp = X2[pos]
        y_temp = Y2[pos]
        X3[pos] = np.cos(angle)*x_temp - np.sin(angle)*y_temp
        Y3[pos] = np.sin(angle)*x_temp + np.cos(angle)*y_temp

    blocks.append([np.reshape(X3,(N,M)),np.reshape(Y3,(N,M))])
    
    bd_point = blocks[0][0][-1,-1]
    x        = np.linspace(-bd_point,bd_point,N)
    y        = np.linspace(-bd_point,-limit,N)
    [X4,Y4]  = np.meshgrid(x,y)
    blocks.append([X4,Y4])

    x = np.linspace(-bd_point,bd_point,N)
    y = np.linspace(limit,bd_point,N)
    [X5,Y5]  = np.meshgrid(x,y)
    blocks.append([X5,Y5])

    grid2d.collocate_corners(blocks)
    bd_info = {
            "bottom_x"  : [-bd_point,bd_point],
            "bottom_y"  : [-limit],
            "right_x"   : [bd_point],
            "right_y"   : [-limit,-bd_point,bd_point,limit],
            "top_x"     : [-bd_point,bd_point],
            "top_y"     : [limit],
            "left_x"    : [-bd_point],
            "left_y"    : [-limit,-bd_point,bd_point,limit],
            "cylinder_x": [blocks[0][0][0,0],blocks[0][0][0,-1], \
                           blocks[2][0][0,0],blocks[2][0][0,-1]],
            "cylinder_y": [blocks[0][1][0,0],blocks[0][1][0,-1], \
                           blocks[2][1][0,0],blocks[2][1][0,-1]]
            }
    
    return blocks, bd_info

def transfinite_quad_map(g1, g2, g3, g4, method):
    
    assert (g1.get_N() == g3.get_N() and g2.get_N() and g4.get_N())

    (x1,y1) = g1.evaluate_at(-1)
    (x2,y2) = g1.evaluate_at(1)
    (x3,y3) = g3.evaluate_at(1)
    (x4,y4) = g3.evaluate_at(-1)

    N       = g1.get_N()
    M       = g2.get_N()
    if method == 'fd':
        xi_vec  = np.linspace(-1,1,N)
        eta_vec = np.linspace(-1,1,M)
    elif method == 'dg':
        xi_vec  = dg_operators.legendre_gauss_lobatto_nodes_and_weights(N-1)[0]
        eta_vec = dg_operators.legendre_gauss_lobatto_nodes_and_weights(M-1)[0]

    x       = np.zeros(N*M)
    y       = np.copy(x)
    pos     = 0

    for i in range(N):
        xi = xi_vec[i]
        (X1,Y1) = g1.evaluate_at(xi)
        (X3,Y3) = g3.evaluate_at(xi)
        for j in range(M):
            eta = eta_vec[j]
            (X2,Y2) = g2.evaluate_at(eta)
            (X4,Y4) = g4.evaluate_at(eta)

            x[pos] = 0.5*((1-xi)*X4 + (1+xi)*X2 + (1-eta)*X1 + (1+eta)*X3) - \
                    0.25*((1-xi)*((1-eta)*x1 + (1+eta)*x4) + \
                          (1+xi)*((1-eta)*x2 + (1+eta)*x3))

            y[pos] = 0.5*((1-xi)*Y4 + (1+xi)*Y2 + (1-eta)*Y1 + (1+eta)*Y3) - \
                    0.25*((1-xi)*((1-eta)*y1 + (1+eta)*y4) + \
                          (1+xi)*((1-eta)*y2 + (1+eta)*y3))
            pos    = pos+1
    X = np.reshape(x,(N,M))
    Y = np.reshape(y,(N,M))
    return X,Y
