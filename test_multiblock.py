import pdb
import numpy as np
from sbpy.utils import get_annulus_grid, create_convergence_table, get_bump_grid
from sbpy.grid2d import MultiblockGrid, MultiblockSBP, get_center

#N = 11
#(X,Y) = np.meshgrid(np.linspace(0,1,N), np.linspace(0,1,N))
#X = np.transpose(X)
#Y = np.transpose(Y)
#grid = MultiblockGrid([(X,Y)])
#sbp = MultiblockSBP(grid, accuracy_x=2, accuracy_y = 2)

#print(sbp.integrate([X*X]))

N = 21
N_vec = [21,41,81,161]
err = []
for N in N_vec:
    blocks = get_annulus_grid(N)
    grid = MultiblockGrid(blocks)
    sbp = MultiblockSBP(grid, accuracy=4)
    
    f = lambda x,y: np.exp(-x**2 - y**2)
    F = []
    for block in blocks:
        F.append(f(block[0],block[1]))
    
    r0 = 0.1
    r1 = 1
    analytic = np.pi*(np.exp(-r0**2) - np.exp(-r1**2))

    err.append(np.abs(sbp.integrate(F) - analytic))

create_convergence_table(N_vec, err, 1/(np.array(N_vec)-1))
N_vec = [161,321,641]

err = []
#for N in N_vec:
##    blocks = get_annulus_grid(N)
#    X,Y = get_bump_grid(N)
#    blocks = [(X,Y),(X+3,Y)]
#    grid = MultiblockGrid(blocks)
#    sbp = MultiblockSBP(grid)
#
#    f = []
#    fp = []
#
#    for block in blocks:
#        f.append(np.sin(2*np.pi*block[0])*np.sin(2*np.pi*block[1]))
#        fp.append(2*np.pi*np.sin(2*np.pi*block[0])*np.cos(2*np.pi*block[1]))
#    df = sbp.diffy(f)
#
#    err_vec = df - fp
#    err.append(np.sqrt(sbp.integrate(err_vec**2)))
#
#create_convergence_table(N_vec, err, 1/(np.array(N_vec)-1))
#
