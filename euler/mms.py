import math
import numpy as np


## mms1
e       = 0.01
k       = 2*np.pi
a       = 0
u       = lambda t,x,y: 1+0.5*np.sin(k*x-a*t)*np.sin(k*y-a*t)
u_t     = lambda t,x,y: -0.5*np.sin(k*x-a*t)*np.cos(k*y-a*t)*a \
                        -0.5*np.cos(k*x-a*t)*np.sin(k*y-a*t)*a
u_x     = lambda t,x,y: 0.5*k*np.cos(k*x-a*t)*np.sin(k*y-a*t)
u_y     = lambda t,x,y: 0.5*k*np.sin(k*x-a*t)*np.cos(k*y-a*t)

u_xx    = lambda t,x,y: -0.5*k*k*np.sin(k*x-a*t)*np.sin(k*y-a*t)
u_yy    = lambda t,x,y: -0.5*k*k*np.sin(k*x-a*t)*np.sin(k*y-a*t)

b       = 0
v       = lambda t,x,y: 1+0.5*np.cos(k*x-b*t)*np.cos(k*y-b*t)
v_t     = lambda t,x,y: 0.5*np.sin(k*x-b*t)*np.cos(k*y-b*t)*b \
                       +0.5*np.cos(k*x-b*t)*np.sin(k*y-b*t)*b
v_x     = lambda t,x,y: -0.5*k*np.sin(k*x-b*t)*np.cos(k*y-b*t)
v_y     = lambda t,x,y: -0.5*k*np.cos(k*x-b*t)*np.sin(k*y-b*t)

v_xx    = lambda t,x,y: -0.5*k*k*np.cos(k*x-b*t)*np.cos(k*y-b*t)
v_yy    = lambda t,x,y: -0.5*k*k*np.cos(k*x-b*t)*np.cos(k*y-b*t)

pi      = np.pi
p       = lambda t,x,y: 0.5*np.cos(2*pi*x)*np.sin(2*pi*y)
p_x     = lambda t,x,y: -pi*np.sin(2*pi*x)*np.sin(2*pi*y)
p_y     = lambda t,x,y: pi*np.cos(2*pi*x)*np.cos(2*pi*y)


############ compute force ##########

uu_x    = lambda t,x,y: 2*u(t,x,y)*u_x(t,x,y)      #(u*u)_x
vv_y    = lambda t,x,y: 2*v(t,x,y)*v_y(t,x,y)      #(v*v)_y

uv_x    = lambda t,x,y: u_x(t,x,y)*v(t,x,y) + u(t,x,y)*v_x(t,x,y)  # (u*v)_x
uv_y    = lambda t,x,y: u_y(t,x,y)*v(t,x,y) + u(t,x,y)*v_y(t,x,y)   


def force1(t,x,y):
    return u_t(t,x,y) + p_x(t,x,y) + \
           0.5*(u(t,x,y)*u_x(t,x,y) + uu_x(t,x,y) + \
                v(t,x,y)*u_y(t,x,y) + uv_y(t,x,y)) - e*(u_xx(t,x,y) + u_yy(t,x,y))

def force2(t,x,y):
    return v_t(t,x,y) + p_y(t,x,y) +\
           0.5*(u(t,x,y)*v_x(t,x,y) + uv_x(t,x,y) + \
                v(t,x,y)*v_y(t,x,y) + vv_y(t,x,y)) - e*(v_xx(t,x,y) + v_yy(t,x,y))

def force3(t,x,y):
    return u_x(t,x,y) + v_y(t,x,y)

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
    ux_bd    = u_x(t,xbd,ybd)
    uy_bd    = u_y(t,xbd,ybd)
    return p_bd*nx - e*(nx*ux_bd + ny*uy_bd)

def tangential_outflow_data(sbp,block_idx,side,t):
    normals = sbp.get_normals(block_idx, side)
    nx   = normals[:,0]
    ny   = normals[:,1]
    xbd, ybd = sbp.grid.get_boundary(block_idx,side)
    u_bd  = u(t,xbd,ybd)
    v_bd  = v(t,xbd,ybd)

    p_bd  = p(t,xbd,ybd)
    vx_bd = v_x(t,xbd,ybd)
    vy_bd = v_y(t,xbd,ybd)

    return p_bd*ny - e*(nx*vx_bd + ny*vy_bd)
