import pdb
import numpy as np
from sbpy.grid2d import MultiblockGrid, MultiblockSBP, collocate_corners

acc           = 4
Nx            = 8
Ny            = 8

blocks = []
x0 = np.linspace(0,1,Nx)
y = np.linspace(0,1,Ny)
[X0,Y0]  = np.meshgrid(x0,y)
X0 = np.transpose(X0)
Y0 = np.transpose(Y0)
blocks.append([X0,Y0])

grid = MultiblockGrid(blocks)
sbp    = MultiblockSBP(grid, accuracy_x = acc, accuracy_y=acc,periodic=False)

print(sbp.get_Dx(0).toarray())

