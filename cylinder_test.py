import numpy as np  
import operators  
import pdb
import utils
import scipy
from sbpy.grid2d import MultiblockGrid, MultiblockSBP, get_center
import matplotlib.pyplot as plt
from matplotlib import cm


N = 60
R = 1.5
r = np.linspace(0,R,N)
th = np.linspace(0,2*np.pi,N)

X,Y = np.meshgrid(r,th)
X = np.transpose(X)
Y = np.transpose(Y)

grid = MultiblockGrid([(X,Y)])
sbp = MultiblockSBP(grid, accuracy=2)

print(sbp.integrate([X**2]) - 2*np.pi*R**3/3)
