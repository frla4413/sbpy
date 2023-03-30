import pdb
import math
import numpy as np

''' mms to be used in the abl-project. The middle region contains the log-law
    u = log(y/beta)/kappa.
    The free-stream function is set to
    u = b1 exp(-b2 y) 
    so that is satisfies the wall condition at y = 0.

    The mms has the following structure:

    y
    ^
    |        ------------------------------------------
    |        |             |            |             |
    |        | free-steram | polynomial | free-steram |
    |        |             |            |             |
    |        ------------------------------------------
    |        |             |            |             |
    |        | free-steram | log-layer  | free-steram |
    |        |             |            |             |
    |        ------------------------------------------
    |        |             |            |             |
    |        | free-steram | polynomial | free-steram |
    |        |             |            |             |
    |        ------------------------------------------
    |
    ---------|-------------|------------|-------------|----> x
            x = 0         x = x0       x = x1

    Window-function:

             | window = 0 |  window=1   |  window=0   |
             | 1-window=1 | 1-window=0  | 1-window=1  |

'''

e       = 0.01
y_lim1  = 1
y_lim2  = 2.5
kappa   = 0.4
beta    = 0.4

a0      = 0
a1      = (6*np.log(y_lim1/beta) - 5)/2/y_lim1/kappa
a2      = (4-3*np.log(y_lim1/beta))/(y_lim1**2)/kappa
a3      = (2*np.log(y_lim1/beta) - 3)/2/(y_lim1**3)/kappa

b0      = 0
b1      = (6*np.log(y_lim2/beta) - 5)/2/y_lim2/kappa
b2      = (4-3*np.log(y_lim2/beta))/(y_lim2**2)/kappa
b3      = (2*np.log(y_lim2/beta) - 3)/2/(y_lim2**3)/kappa

s1   = 5
s2   = np.pi/2
h    = lambda x: 0.5*(1 + np.arctan(x*s1)/s2)
h_x  = lambda x: 0.5*s1/((x*s1)**2 + 1)/s2
h_xx = lambda x: -s1**3*x/((s1*x)**2 + 1)**2/s2

x0 = 0.5
x1 = 1.5
window    = lambda x: h(x - x0) - h(x - x1)
window_x  = lambda x: h_x(x - x0) - h_x(x - x1)
window_xx = lambda x: h_xx(x - x0) - h_xx(x - x1)

d1 = 4
d2 = -1
u_free_stream    = lambda y: d1*(1 - np.exp(d2*y))
u_free_stream_y  = lambda y: -d1*d2*np.exp(d2*y)
u_free_stream_yy = lambda y: -d1*d2*d2*np.exp(d2*y)

def u_mix_layer(y):
    out = []
    for yi in y.flatten():
        if yi <= y_lim1:
            u1 = a0 + a1*yi + a2*yi**2 + a3*yi**3
        elif yi > y_lim1 and yi <= y_lim2:
            u1 = np.log(yi/beta)/kappa
        else:
            u1 = b0 + b1*yi + b2*yi**2 + b3*yi**3
        out.append(u1)
    return np.reshape(out,y.shape)

def u_mix_layer_y(y):
    out = []
    for yi in y.flatten():
        if yi <= y_lim1:
            u1_y = a1 + 2*a2*yi + 3*a3*yi**2
        elif yi > y_lim1 and yi <= y_lim2:
            u1_y = 1/yi/kappa
        else:
            u1_y = b1 + 2*b2*yi + 3*b3*yi**2
        out.append(u1_y)
    return np.reshape(out,y.shape)

def u_mix_layer_yy(y):
    out = []
    for yi in y.flatten():
        if yi <= y_lim1:
            u1_yy = 2*a2 + 6*a3*yi
        elif yi > y_lim1 and yi <= y_lim2:
            u1_yy = -1/(yi**2)/kappa
        else:
            u1_yy = 2*b2 + 6*b3*yi
        out.append(u1_yy)
    return np.reshape(out,y.shape)

def u(t,x,y):
    out1 = u_mix_layer(y)*window(x)
    out2 = u_free_stream(y)*(1-window(x))
    return out1 + out2
           
def u_x(t,x,y):
    out1 = u_mix_layer(y)*window_x(x)
    out2 = -u_free_stream(y)*window_x(x)
    return out1 + out2

def u_y(t,x,y):
    out1 = u_mix_layer_y(y)*window(x)
    out2 = u_free_stream_y(y)*(1-window(x))
    return out1 + out2

def u_xx(t,x,y):
    out1 = u_mix_layer(y)*window_xx(x)
    out2 = -u_free_stream(y)*window_xx(x)
    return out1 + out2

def u_yy(t,x,y):
    out1 = u_mix_layer_yy(y)*window(x)
    out2 = u_free_stream_yy(y)*(1-window(x))
    return out1 + out2

u_t = lambda t,x,y: 0

k  = 0.5*np.pi

v   = lambda t,x,y: np.sin(k*x)*np.sin(k*y)
v_t = lambda t,x,y: 0
v_x = lambda t,x,y: k*np.cos(k*x)*np.sin(k*y)
v_y = lambda t,x,y: k*np.sin(k*x)*np.cos(k*y)

v_xx= lambda t,x,y: -k*k*np.sin(k*x)*np.sin(k*y)
v_yy= lambda t,x,y: -k*k*np.sin(k*x)*np.sin(k*y)

p   = lambda t,x,y: np.cos(k*x)*np.cos(k*y)
p_x = lambda t,x,y: -k*np.sin(k*x)*np.cos(k*y)
p_y = lambda t,x,y: -k*np.cos(k*x)*np.sin(k*y)


############ compute force ##########

uu_x = lambda t,x,y: 2*u(t,x,y)*u_x(t,x,y)      #(u*u)_x
vv_y = lambda t,x,y: 2*v(t,x,y)*v_y(t,x,y)      #(v*v)_y

uv_x = lambda t,x,y: u_x(t,x,y)*v(t,x,y) + u(t,x,y)*v_x(t,x,y)  # (u*v)_x
uv_y = lambda t,x,y: u_y(t,x,y)*v(t,x,y) + u(t,x,y)*v_y(t,x,y)   


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
