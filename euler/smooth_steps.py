import numpy as np
import matplotlib.pyplot as plt
import pdb
from sbpy.grid2d import MultiblockGrid, get_boundary


linewidth = 2.5
x = np.linspace(0,1,13)

a = 2
b = (1-x)**(a)
y = x**a/(x**a + b)
y = y*2
X,Y = np.meshgrid(np.linspace(0,1,5), y)

grid = MultiblockGrid([(X,Y)])

plt.rcParams["figure.figsize"] = (10,8)
font = {'size' : 20}
plt.rc('font', **font)

plt.plot(X,Y,'b',linewidth=linewidth)
plt.plot(np.transpose(X),np.transpose(Y),'b',linewidth=linewidth)

for side in {'w','e','s','n'}:
    x,y = get_boundary(X,Y, side)
    plt.plot(x,y,'k',linewidth=linewidth)
plt.xticks([0, 0.5, 1])
plt.yticks([0, 1, 2])
plt.show()

