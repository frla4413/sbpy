import math
import numpy as np


## mms1
k       = 2
a       = 0
pi      = math.pi
u       = lambda t,x,y: 1+0.5*np.sin(k*pi*x-a*t)*np.sin(k*pi*y-a*t)
u_t     = lambda t,x,y: -0.5*np.sin(k*pi*x-a*t)*np.cos(k*pi*y-a*t)*a \
                        -0.5*np.cos(k*pi*x-a*t)*np.sin(k*pi*y-a*t)*a
u_x     = lambda t,x,y: 0.5*k*pi*np.cos(k*pi*x-a*t)*np.sin(k*pi*y-a*t)
u_y     = lambda t,x,y: 0.5*k*pi*np.sin(k*pi*x-a*t)*np.cos(k*pi*y-a*t)

b       = 0
v       = lambda t,x,y: 1+0.5*np.cos(k*pi*x-b*t)*np.cos(k*pi*y-b*t)
v_t     = lambda t,x,y: 0.5*np.sin(k*pi*x-b*t)*np.cos(k*pi*y-b*t)*b \
                       +0.5*np.cos(k*pi*x-b*t)*np.sin(k*pi*y-b*t)*b
v_x     = lambda t,x,y: -pi*np.sin(k*pi*x-b*t)*np.cos(k*pi*y-b*t)
v_y     = lambda t,x,y: -pi*np.cos(k*pi*x-b*t)*np.sin(k*pi*y-b*t)

p       = lambda t,x,y: 0.5*np.cos(2*pi*x)*np.sin(2*pi*y)
p_x     = lambda t,x,y: -pi*np.sin(2*pi*x)*np.sin(2*pi*y)
p_y     = lambda t,x,y: pi*np.cos(2*pi*x)*np.cos(2*pi*y)

## mms2
#pi      = math.pi
#u       = lambda t,x,y: pi*np.ones(x.shape)
#u_t     = lambda t,x,y: 0
#u_x     = lambda t,x,y: np.zeros(x.shape)
#u_y     = lambda t,x,y: np.zeros(x.shape)
#
#v       = lambda t,x,y: pi*np.ones(x.shape)
#v_t     = lambda t,x,y: np.zeros(x.shape)
#v_x     = lambda t,x,y: np.zeros(x.shape)
#v_y     = lambda t,x,y: np.zeros(x.shape)
#
#p       = lambda t,x,y: np.zeros(x.shape)
#p_x     = lambda t,x,y: np.zeros(x.shape)
#p_y     = lambda t,x,y: np.zeros(x.shape)

############ compute force ##########

uu_x    = lambda t,x,y: 2*u(t,x,y)*u_x(t,x,y)      #(u*u)_x
vv_y    = lambda t,x,y: 2*v(t,x,y)*v_y(t,x,y)      #(v*v)_y

uv_x    = lambda t,x,y: u_x(t,x,y)*v(t,x,y) + u(t,x,y)*v_x(t,x,y)  # (u*v)_x
uv_y    = lambda t,x,y: u_y(t,x,y)*v(t,x,y) + u(t,x,y)*v_y(t,x,y)   


def force1(t,x,y):
    return u_t(t,x,y) + p_x(t,x,y) + \
           0.5*(u(t,x,y)*u_x(t,x,y) + uu_x(t,x,y) + \
                v(t,x,y)*u_y(t,x,y) + uv_y(t,x,y))

def force2(t,x,y):
    return v_t(t,x,y) + p_y(t,x,y) +\
           0.5*(u(t,x,y)*v_x(t,x,y) + uv_x(t,x,y) + \
                v(t,x,y)*v_y(t,x,y) + vv_y(t,x,y))

def force3(t,x,y):
    return u_x(t,x,y) + v_y(t,x,y)

def wn_data(sbp, block_idx, side, t):
    n        = sbp.get_normals(block_idx,side)
    nx       = n[:,0] 
    ny       = n[:,1] 
    xbd, ybd = sbp.grid.get_boundary(block_idx,side)
    return u(t,xbd,ybd)*nx + v(t,xbd,ybd)*ny

def wt_data(sbp, block_idx, side, t):
    n        = sbp.get_normals(block_idx,side)
    nx       = n[:,0] 
    ny       = n[:,1] 
    xbd, ybd = sbp.grid.get_boundary(block_idx,side)
    return - u(t,xbd,ybd)*ny + v(t,xbd,ybd)*nx

def p_data(sbp, block_idx, side, t):
    xbd, ybd = sbp.grid.get_boundary(block_idx,side)
    return p(t,xbd,ybd)

