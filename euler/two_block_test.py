import pdb
import numpy as np
from sbpy.grid2d import MultiblockGrid, MultiblockSBP, flatten_multiblock_vector
from sbpy.utils import create_convergence_table, solution_to_file
from sbpy.meshes import set_bd_info, get_cylinder_channel_grid
from euler import solve_euler_ibvp, solve_euler_steady_state, jans_inflow_operator
from sbpy.grid2d import load_p3d

acc           = 4
dt            = 1e-2
num_timesteps = 70
Nx            = 75
Ny            = 40

blocks = []
x = np.linspace(-1,0,Nx)
y = np.linspace(0,1,Ny)
[X0,Y0]  = np.meshgrid(x,y)
X0 = np.transpose(X0)
Y0 = np.transpose(Y0)
X0 = X0[:,:-1]
Y0 = Y0[:,:-1]
blocks.append([X0,Y0])

x = np.linspace(0,1,Nx)
[X1,Y1]  = np.meshgrid(x,y)
X1 = np.transpose(X1)
Y1 = np.transpose(Y1)
X1 = X1[:,:-1]
Y1 = Y1[:,:-1]
blocks.append([X1,Y1])

x = np.linspace(1,2,Nx)
[X2,Y2]  = np.meshgrid(x,y)
X2 = np.transpose(X2)
Y2 = np.transpose(Y2)
X2 = X2[:,:-1]
Y2 = Y2[:,:-1]
blocks.append([X2,Y2])

grid = MultiblockGrid(blocks)

#grid.plot_domain(boundary_indices=True)
#print(grid.get_boundaries())
#for i in range(len(grid.get_interfaces())):
#    if grid.is_flipped_interface(i):
#        print('flipped interfaces: ' + str(i))
#grid.plot_grid()
sbp = MultiblockSBP(grid, accuracy_x = acc, accuracy_y=2,periodic=True)

x0 = -1
x1 = 1
x2 = 2
y0 = y[0]
y1 = y[-2]
bd_info = {
        "bd1_x": [-1,0,1,2],
        "bd1_y": [y0],
        "bd2_x": [x2],
        "bd2_y": [y0,y1],
        "bd3_x": [x0,0,x1,x2],
        "bd3_y": [y1],
        "bd4_x": [x0],
        "bd4_y": [y0,y1]
        }

boundary_condition = {
        "bd1" : "periodic",
        "bd2" : "pressure",
        "bd3" : "periodic",
        "bd4" : "inflow"
        }

set_bd_info(grid, bd_info, boundary_condition)
initu = []
initv = []
initp = []

for (X,Y) in blocks:
    initu.append(np.ones(X.shape))
    initv.append(np.zeros(X.shape))
    initp.append(np.zeros(X.shape))

def window(x,y, xc, yc):
    jump = 0.2
    out = 0.5*(np.tanh(50*(x-xc+jump)) - np.tanh(50*(x-xc-jump)))
    return out

for k,(X,Y) in enumerate(blocks):
    initu[k] += 0.1*np.sin(2*np.pi*X)*np.cos(2*np.pi*Y)*window(X,Y,-0.25,0.5)
    initv[k] += -0.1*np.cos(2*np.pi*X)*np.sin(2*np.pi*Y)*window(X,Y,-0.25,0.5)


div = sbp.diffx(initu) + sbp.diffy(initv)
print(np.linalg.norm(flatten_multiblock_vector(div),ord=np.inf))

gu = 1#initu[0][0] - 0.05
gv = 0

def wn_data(sbp, block_idx, side,t):
    normals = sbp.get_normals(block_idx, side)
    nx      = normals[:,0]
    ny      = normals[:,1]
    return gu*nx + gv*ny

def wt_data(sbp, block_idx, side,t):
    normals = sbp.get_normals(block_idx, side)
    nx      = normals[:,0]
    ny      = normals[:,1]
    return -gu*ny + gv*nx

bd_data = {
    "wn_data": wn_data,
    "wt_data": wt_data,
    "p_data" : 0,
    "jans_data": gu
    }

name = 'long_domain/plots_interface/Nx75/sol'
solve_euler_ibvp(sbp, boundary_condition, bd_data, initu, initv, initp, \
                 dt, num_timesteps, name_base = name)
