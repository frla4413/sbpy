import numpy as np
import pdb
import matplotlib.pyplot as plt

Nx = 75
x1 = np.linspace(-1,2,3*Nx-2)
x2 = np.zeros((3*Nx,))
x2[:Nx] = np.linspace(-1,0,Nx)
x2[Nx:2*Nx] = np.linspace(0,1,Nx) 
x2[2*Nx:] = np.linspace(1,2,Nx) 

out = np.zeros((3*Nx-2,))
out[:Nx-1] = x2[:Nx-1]
out[Nx-1] = 0.5*(x2[Nx-1] + x2[Nx])
out[Nx:2*Nx-1] = x2[Nx+1:2*Nx]
out[2*Nx-1] = 0.5*(x2[2*Nx-1] + x2[2*Nx])
out[2*Nx-1:] = x2[2*Nx+1:]

err = x1 - out

plt.plot(x1,err)
plt.show()
pdb.set_trace()

