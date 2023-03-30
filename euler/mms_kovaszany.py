import numpy as np

e = 1/20
pi= np.pi
l = 1/2/e - np.sqrt(1/4/e/e + 4*pi**2)

u  = lambda t,x,y: 1-np.exp(l*x)*np.cos(2*pi*y)
ux = lambda t,x,y:-l*np.exp(l*x)*np.cos(2*pi*y)
uy = lambda t,x,y: 2*pi*np.exp(l*x)*np.sin(2*pi*y)

v  = lambda t,x,y: l*np.exp(l*x)*np.sin(2*pi*y)*0.5/pi
vx = lambda t,x,y: l**2*np.exp(l*x)*np.sin(2*pi*y)*0.5/pi
vy = lambda t,x,y: l*np.exp(l*x)*np.cos(2*pi*y)

p  = lambda t,x,y: (1-np.exp(2*l*x))/2


def wn_data(sbp, block_idx, side, t):
    normals = sbp.get_normals(block_idx, side)
    nx = normals[:,0]
    ny = normals[:,1]
    bd_slice = sbp.grid.get_boundary_slice(block_idx, side)
    xbd, ybd = sbp.grid.get_boundary(block_idx,side)
    u_bd = u(t,xbd,ybd)
    v_bd = v(t,xbd,ybd)

    return nx*u_bd + ny*v_bd

def wt_data(sbp, block_idx, side, t):
    normals = sbp.get_normals(block_idx, side)
    nx = normals[:,0]
    ny = normals[:,1]
    bd_slice = sbp.grid.get_boundary_slice(block_idx, side)
    xbd, ybd = sbp.grid.get_boundary(block_idx,side)
    u_bd = u(t,xbd,ybd)
    v_bd = v(t,xbd,ybd)

    return -ny*u_bd + nx*v_bd

def normal_outflow_data(sbp, block_idx, side, t):
    normals = sbp.get_normals(block_idx, side)
    nx      = normals[:,0]
    ny      = normals[:,1]
    xbd, ybd = sbp.grid.get_boundary(block_idx,side)
    u_bd = u(t,xbd,ybd)
    v_bd = v(t,xbd,ybd)

    p_bd     = p(t,xbd,ybd)
    ux_bd    = ux(t,xbd,ybd)
    uy_bd    = uy(t,xbd,ybd)
    return p_bd*nx - e*(nx*ux_bd + ny*uy_bd)

def tangential_outflow_data(sbp,block_idx,side,t):
    normals = sbp.get_normals(block_idx, side)
    nx   = normals[:,0]
    ny   = normals[:,1]
    xbd, ybd = sbp.grid.get_boundary(block_idx,side)
    u_bd = u(t,xbd,ybd)
    v_bd = v(t,xbd,ybd)

    p_bd     = p(t,xbd,ybd)
    vx_bd    = vx(t,xbd,ybd)
    vy_bd    = vy(t,xbd,ybd)

    return p_bd*ny - e*(nx*vx_bd + ny*vy_bd)
