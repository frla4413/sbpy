""" This module contains functions for displaying and interacting with various
objects. """

import itertools
import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import widgets
from matplotlib import path
from matplotlib import patches
from sbpy import grid2d
from sbpy import utils


class NodeSelector:
    """ A class for visually selecting nodes from a MultiblockGrid object.
    Initialize with a MultiblockGrid object, NodeSelector(grid). A NodeSelector is
    callable--run it without argument to show the selection window. Once you have
    selected your nodes, they will be stored in the nodes list. """

    def __init__(self, grid):
        self.grid = grid
        self.nodes = []
        self.fig, self.ax = plt.subplots()
        self.x_points = []
        self.y_points = []
        for (X,Y) in self.grid.get_blocks():
            self.x_points.append(X.flatten())
            self.y_points.append(Y.flatten())

        self.x_points = np.concatenate(self.x_points)
        self.y_points = np.concatenate(self.y_points)
        colors = [[0,0,1] for _ in range(len(self.x_points))]
        self.ax.scatter(self.x_points, self.y_points, c=colors, picker=5)

        self.pt_collection = self.ax.collections[0]
        self.ax.axis('equal')

    def _flat_to_multi_idx(self, idx):
        block = 0
        count = 0
        shapes = self.grid.get_shapes()
        for (k,shape) in enumerate(shapes):
            count += shape[0]*shape[1]
            if idx < count:
                block = k
                break

        local_flat_idx = idx - (count-shapes[k][0]*shapes[k][1])
        Nx,Ny = shapes[k]
        (i,j) = (int(np.floor(local_flat_idx/Ny)),int(local_flat_idx%Ny))
        return (block,i,j)


    def __call__(self):
        #lasso = widgets.LassoSelector(self.ax, self._onselect)
        ellipse = widgets.EllipseSelector(self.ax, self._onselect)
        self.fig.canvas.mpl_connect('pick_event', self._onpick)
        plt.show()


    def _onpick(self, event):
        fc = self.pt_collection.get_facecolors()
        multi_idx = [self._flat_to_multi_idx(i) for i in event.ind]

        for multi,flat in zip(multi_idx,event.ind):
            if multi in self.nodes:
                fc[event.ind] = [0,0,1,1]
                self.nodes = [ node for node in self.nodes if node != multi ]
            else:
                fc[event.ind] = [1,0,0,1]
                self.nodes.append(multi)

        self.ax.figure.canvas.draw_idle()

    #def _onselect(self, verts):
    #    p = path.Path(verts)
    #    fc = self.pt_collection.get_facecolors()
    #    pts = [(x,y) for x,y in zip(self.x_points, self.y_points)]
    #    ind = np.nonzero(p.contains_points(pts))[0]
    #    multi_idx = [self._flat_to_multi_idx(i) for i in ind]

    #    for multi,flat in zip(multi_idx,ind):
    #        if multi not in self.nodes:
    #            fc[ind] = [1,0,0,1]
    #            self.nodes.append(multi)

    #    self.ax.figure.canvas.draw_idle()
    #    print(self.nodes)

    def _onselect(self, eclick, erelease):
        x1,y1 = (eclick.xdata, eclick.ydata)
        x2,y2 = (erelease.xdata, erelease.ydata)
        mid = (0.5*(x1+x2), 0.5*(y1+y2))
        w = np.abs(x2-x1)
        h = np.abs(y2-y1)
        p = patches.Ellipse(mid,w,h)
        fc = self.pt_collection.get_facecolors()
        pts = [(x,y) for x,y in zip(self.x_points, self.y_points)]
        ind = np.nonzero(p.contains_points(pts))[0]
        multi_idx = [self._flat_to_multi_idx(i) for i in ind]

        for multi,flat in zip(multi_idx,ind):
            if multi not in self.nodes:
                fc[ind] = [1,0,0,1]
                self.nodes.append(multi)

        self.ax.figure.canvas.draw_idle()
        print(self.nodes)


if __name__ == '__main__':
    N = 21
    blocks = [utils.get_circle_sector_grid(N, 0, 0.5*np.pi, 0.2, 1.0),
              utils.get_circle_sector_grid(N, 0.5*np.pi, np.pi, 0.2, 1.0),
              utils.get_circle_sector_grid(N, np.pi, 1.5*np.pi, 0.2, 1.0),
              utils.get_circle_sector_grid(N, 1.5*np.pi, 2*np.pi, 0.2, 1.0)]
    grid2d.collocate_corners(blocks)
    coarse_grid = grid2d.MultiblockGridSBP(blocks, accuracy=4)
    selector = NodeSelector(coarse_grid)
    selector()


    N = 41
    blocks = [utils.get_circle_sector_grid(N, 0, 0.5*np.pi, 0.2, 1.0),
              utils.get_circle_sector_grid(N, 0.5*np.pi, np.pi, 0.2, 1.0),
              utils.get_circle_sector_grid(N, np.pi, 1.5*np.pi, 0.2, 1.0),
              utils.get_circle_sector_grid(N, 1.5*np.pi, 2*np.pi, 0.2, 1.0)]
    grid2d.collocate_corners(blocks)
    fine_grid = grid2d.MultiblockGridSBP(blocks, accuracy=4)

    F = [ X + Y for X,Y in fine_grid.get_blocks() ]

    dat,ind = utils.fetch_highres_data(coarse_grid, selector.nodes, fine_grid, F)
    print(ind)


