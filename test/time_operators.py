import numpy as np
import pdb
import time
import sbpy.operators
from sbpy import grid2d
import scipy

N = 2000
x = np.linspace(0, 1, N)
dx = x[1] - x[0]
sbp_op = sbpy.operators.SBP1D(N, dx)

sbp_op.D@x

E0 = scipy.sparse.csr_matrix(sbp_op.D.shape)

E0[0,0] = 1
D_tilde = sbp_op.D  + E0*N

b = D_tilde*x

sol = scipy.sparse.linalg.gmres(D_tilde,b)

err = np.linalg.norm(sol[0] - x)
print(err)
