import pdb
import numpy as np
from sbpy.dg_operators import DGSBP1D,legendre_gauss_lobatto_nodes_and_weights, CurveInterpolant
import matplotlib.pyplot as plt
from sbpy.meshes import transfinite_quad_map, get_cylinder_channel_grid
from sbpy.grid2d import MultiblockGrid, collocate_corners

radius = 1.2
N      = 20
blocks, bd_info = get_cylinder_channel_grid(N,N,radius)
bd_point = blocks[0][0][-1,-1]

limit = 5
x = np.linspace(bd_point,limit,N)
y = np.linspace(-bd_point,bd_point,N)
[X,Y]  = np.meshgrid(x,y)
blocks.append([X,Y])

x = np.linspace(-bd_point,bd_point,N)
y = np.linspace(-bd_point,-limit,N)
[X,Y]  = np.meshgrid(x,y)
blocks.append([X,Y])

x = np.linspace(bd_point,limit,N)
y = np.linspace(-bd_point,-limit,N)
[X,Y]  = np.meshgrid(x,y)
blocks.append([X,Y])

x = np.linspace(-bd_point,bd_point,N)
y = np.linspace(bd_point,limit,N)
[X,Y]  = np.meshgrid(x,y)
blocks.append([X,Y])

x = np.linspace(bd_point,limit,N)
y = np.linspace(bd_point,limit,N)
[X,Y]  = np.meshgrid(x,y)
blocks.append([X,Y])

x = np.linspace(-bd_point,-limit,N)
y = np.linspace(-bd_point,bd_point,N)
[X,Y]  = np.meshgrid(x,y)
blocks.append([X,Y])

x = np.linspace(-bd_point,-limit,N)
y = np.linspace(-bd_point,-limit,N)
[X,Y]  = np.meshgrid(x,y)
blocks.append([X,Y])

x = np.linspace(-bd_point,-limit,N)
y = np.linspace(bd_point,limit,N)
[X,Y]  = np.meshgrid(x,y)
blocks.append([X,Y])


collocate_corners(blocks)
grid   = MultiblockGrid(blocks)
#grid.plot_domain(boundary_indices=True)
grid.plot_grid()
