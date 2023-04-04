from sbpy import operators
import numpy as np
from numba import jit
import utils
import pdb
import scipy
import matplotlib.pyplot as plt

def init(z,k,height):
    r = z - height
    x = 0
    w0 = 1
    wz = 10
    R = 1
    return (w0/wz)*np.exp(-r**2/wz**2)*np.exp(-1j*(k*x + k*r**2/2/R))

z0 = 0
z1 = 250
N = 5*250
height = 50
acc = 8

z = np.linspace(z0,z1,N)
dz = z[1] - z[0]

lam = 1
k = 2*np.pi/lam
n2 = 1.5*np.ones(z.shape) #+ 1j*(1 + 0.5*np.tanh((z-200)/20))
a = 0.5/k
b = 0.5*k*(n2-1.0)

alpha = k*np.sqrt(n2[0]-1)/n2[0]
beta = 1j/a

sbp = operators.SBP1D(N,dz,acc)
I = np.eye(N)
D2 = sbp.D@sbp.D
E0 = np.zeros(N)
E0[0] = 1
E0 = scipy.sparse.diags(E0)

EN = np.zeros(N)
EN[-1] = 1
EN = scipy.sparse.diags(EN)

mat = 1j*(a*D2 + b@I)
SAT = 1j*a*sbp.P_inv@(E0@(sbp.D + alpha*I) - EN@(sbp.D - beta*I))
rhs = mat + SAT

y0 = init(z,k,height)

@jit(nopython=True)
def fun(t,u):
    return np.reshape(np.asarray(rhs@u),(N,))

t1 = 1500
times = np.linspace(0,t1,201)
sol = scipy.integrate.solve_ivp(fun,[0,t1], y0, rtol=1e-6,atol=1e-6,t_eval=times)

[Z,X] = np.meshgrid(sol.t,z)
Y = np.abs(sol.y)

utils.export_to_tecplot(Z,X,Y,'bla3')
