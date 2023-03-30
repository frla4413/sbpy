""" This module contains visualization tools for PDE solutions. """

import itertools
import numpy as np
from mayavi import mlab


def animate_multiblock(grid, F, **kwargs):
    """ Animates a list of multiblock functions.

    Arguments:
        F: A list of multiblock functions.

    Optional:
        fps: A positive integer representing the number of frames per second
        stored in F.
        stride: A positive integer representing the stride length in the data.
            By default stride = 1, but for high resolution grids consider
            increasing the stride for efficient rendering.

    """

    if 'fps' in kwargs:
        fps = kwargs['fps']
    else:
        fps = 30

    if 'stride' in kwargs:
        stride = kwargs['stride']
    else:
        stride = 1

    Fmin = np.min(np.array(F))
    Fmax = np.max(np.array(F))
    xmin = np.min(np.array([X for X,Y in grid.get_blocks()]))
    xmax = np.max(np.array([X for X,Y in grid.get_blocks()]))
    ymin = np.min(np.array([Y for X,Y in grid.get_blocks()]))
    ymax = np.max(np.array([Y for X,Y in grid.get_blocks()]))
    surfaces = [ mlab.mesh(X[::stride, ::stride],
                           Y[::stride, ::stride],
                           Z[::stride, ::stride],
                           vmax = Fmax, vmin = Fmin) for ((X,Y),Z) in
                 zip(grid.get_blocks(), F[0]) ]


    @mlab.animate(delay=int(1000/fps))
    def anim():
        for f in itertools.cycle(F):
            for (s,f_block) in zip(surfaces, f):
                s.mlab_source.trait_set(scalars=f_block[::stride, ::stride])
                s.mlab_source.trait_set(z=f_block[::stride, ::stride])
            yield

    mlab.axes(x_axis_visibility = False,
              y_axis_visibility = False,
              z_axis_visibility = False,
              extent=[xmin,xmax,ymin,ymax,Fmin,Fmax])

    frame_gen = anim()
    mlab.show()
