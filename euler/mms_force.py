import math
import numpy as np


## mms1
a       = 0.00
pi      = math.pi
u       = lambda t,x,y: 1+0.5*np.sin(2*pi*x)*np.sin(2*pi*y-a*t)
u_t     = lambda t,x,y: -0.5*np.sin(2*pi*x)*np.cos(2*pi*y-a*t)*a
u_x     = lambda t,x,y: pi*np.cos(2*pi*x)*np.sin(2*pi*y-a*t)
u_y     = lambda t,x,y: pi*np.sin(2*pi*x)*np.cos(2*pi*y-a*t)

v       = lambda t,x,y: 1+0.5*np.cos(2*pi*x)*np.cos(2*pi*y)
v_t     = lambda t,x,y: 0
v_x     = lambda t,x,y: -pi*np.sin(2*pi*x)*np.cos(2*pi*y)
v_y     = lambda t,x,y: -pi*np.cos(2*pi*x)*np.sin(2*pi*y)

p       = lambda t,x,y: 0.5*np.cos(2*pi*x)*np.sin(2*pi*y)
p_x     = lambda t,x,y: -pi*np.sin(2*pi*x)*np.sin(2*pi*y)
p_y     = lambda t,x,y: pi*np.cos(2*pi*x)*np.cos(2*pi*y)

## mms2
#pi      = math.pi
#u       = lambda t,x,y: 1+0.5*np.sin(2*pi*x)*np.sin(2*pi*y) + 0.01*t
#u_t     = lambda t,x,y: 0.01
#u_x     = lambda t,x,y: pi*np.cos(2*pi*x)*np.sin(2*pi*y)
#u_y     = lambda t,x,y: pi*np.sin(2*pi*x)*np.cos(2*pi*y)
#
#v       = lambda t,x,y: 1+0.5*np.cos(2*pi*x)*np.cos(2*pi*y) + 0.01*t
#v_t     = lambda t,x,y: 0.01
#v_x     = lambda t,x,y: -pi*np.sin(2*pi*x)*np.cos(2*pi*y)
#v_y     = lambda t,x,y: -pi*np.cos(2*pi*x)*np.sin(2*pi*y)
#
#p       = lambda t,x,y: 0.5*np.cos(2*pi*x)*np.sin(2*pi*y) + 0.01*t
#p_x     = lambda t,x,y: -pi*np.sin(2*pi*x)*np.sin(2*pi*y)
#p_y     = lambda t,x,y: pi*np.cos(2*pi*x)*np.cos(2*pi*y)

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
