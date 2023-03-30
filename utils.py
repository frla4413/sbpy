""" This module contains various utility functions. """

import itertools
import numpy as np
from sbpy import grid2d

def create_convergence_table(labels, errors, h, title=None, filename=None):
    """
    Creates a latex convergence table.
    Parameters:
      labels: An array strings describing each grid (e.g. number of nodes).
      errors: An array of errors.
      h: An array of grid sizes.

    Optional Parameters:
      title: Table title.
      filename: Write table to file.

    Output:
      Prints tex code for the table.
    """

    errors = np.array(errors)
    h = np.array(h)

    rates = (np.log(errors[:-1]) - np.log(errors[1:]))/(np.log(h[:-1]) - np.log(h[1:]))

    N = len(errors)
    print("\\begin{tabular}{|l|l|l|}")
    print("\hline")

    if title:
        print("\multicolumn{{3}}{{|c|}}{{{}}} \\\\".format(title))

    print("\hline")
    print("& error & rate \\\\".format(errors[0]))
    print("\hline")
    print(labels[0], " & {:.4e} & - \\\\".format(errors[0]))
    for k in range(1,N):
        print(labels[k], " & {:.4e} & {:.2f} \\\\".format(errors[k], rates[k-1]))
    print("\hline")
    print("\\end{tabular}")

    if filename:
        with open(filename,'a') as f:
            f.write("\\begin{tabular}{|l|l|l|}\n")
            f.write("\hline\n")

            if title:
                f.write("\multicolumn{{3}}{{|c|}}{{{}}} \\\\\n".format(title))

            f.write("\hline\n")
            f.write("& error & rate \\\\\n".format(errors[0]))
            f.write("\hline\n")
            f.write(str(labels[0]) + " & {:.4e} & - \\\\\n".format(errors[0]))
            for k in range(1,N):
                f.write(str(labels[k]) + " & {:.4e} & {:.2f} \\\\\n".format(errors[k], rates[k-1]))
            f.write("\hline\n")
            f.write("\\end{tabular}\n\n")


def get_circle_sector_grid(N, th0, th1, r_inner, r_outer):
    """ Returns a circle sector grid.

    Arguments:
        N: Number of gridpoints in each direction.
        th0: Start angle.
        th1: End angle.
        r_inner: Inner radius.
        r_outer: Outer radius.

    Returns:
        (X,Y): A pair of matrices defining the grid.
    """
    d_r = (r_outer - r_inner)/(N-1)
    d_th = (th1-th0)/(N-1)

    radii = np.linspace(r_inner, r_outer, N)
    thetas = np.linspace(th0, th1, N)

    x = np.zeros(N*N)
    y = np.zeros(N*N)

    pos = 0
    for r in radii:
        for th in thetas:
            x[pos] = r*np.cos(th)
            y[pos] = r*np.sin(th)
            pos += 1

    X = np.reshape(x,(N,N))
    Y = np.reshape(y,(N,N))

    return X,Y


def get_annulus_grid(N):
    """ Returns a list of four blocks constituting an annulus grid. """
    blocks = [get_circle_sector_grid(N, 0, 0.5*np.pi, 0.2, 1.0),
              get_circle_sector_grid(N, 0.5*np.pi, np.pi, 0.2, 1.0),
              get_circle_sector_grid(N, np.pi, 1.5*np.pi, 0.2, 1.0),
              get_circle_sector_grid(N, 1.5*np.pi, 2*np.pi, 0.2, 1.0)]
    grid2d.collocate_corners(blocks)
    return blocks


def get_bump_grid(N):
    """ Returns a grid with two bumps in the floor and ceiling.
    Arguments:
        N: Number of gridpoints in each direction.

    Returns:
        (X,Y): A pair of matrices defining the grid.
    """
    x0 = -1.5
    x1 = 1.5
    dx = (x1-x0)/(N-1)
    y0 = lambda x: 0.0625*np.exp(-25*x**2)
    y1 = lambda y: 0.8
    x = np.zeros(N*N)
    y = np.copy(x)
    pos = 0
    for i in range(N):
        for j in range(N):
            x_val = x0 + i*dx
            x[pos] = x_val
            y[pos] = y0(x_val) + j*(y1(x_val)-y0(x_val))/(N-1)
            pos = pos+1

    X = np.reshape(x,(N,N))
    Y = np.reshape(y,(N,N))

    return X,Y


def fetch_highres_data(coarse_grid, coarse_indices, fine_grid,
                       fine_data, stride):
    """ Retrieves data from a high resolution grid to be used in a low resolution
    grid.

    Arguments:
        coarse_grid: A Multiblock object representing the coarse grid.
        coarse_indices: A list of indices of the form (block_idx, i, j) where we
            want to fetch data from the fine grid.
        fine_grid: A Multiblock object representing the fine grid.
        fine_data: A multiblock function on the fine grid.
        stride: The stride in the fine grid that gives the coarse grid. I.e.
            if (X,Y) is a block in the fine grid, then X[::stride], Y[::stride]
            is the same block in the coarse grid.

    Returns:
        coarse_data: Function evaluations fetched from the fine grid.
        coarse_indices: A list of indices in the coarse grid corresponding to the
            fetched data.
    """

    coarse_data = []
    for (blk,ic,jc) in coarse_indices:
        coarse_data.append(fine_data[blk][ic*stride,jc*stride])

    return coarse_data


def boundary_layer_selection(grid, bd_indices, n):
    """ Returns a list of indices of the n closest slices of interior nodes to
        the supplied boundaries. Use grid.plot_domain(boundary_indices=True) to
        view the boundary indices of your grid.

    Arguments:
        grid: A Multiblock object
        bd_indices: A list of boundary indices.
        n: The number of interior nodes orthogonal to the boundaries to select.
    """
    boundaries = grid.get_boundaries()
    shapes = grid.get_shapes()
    indices = []
    for bd_idx in bd_indices:
        blk_idx, side = boundaries[bd_idx]
        (Nx, Ny) = shapes[blk_idx]

        if side == 'w':
            indices = indices + [ (blk_idx, i, j) for i,j in
                        itertools.product(range(n), range(Ny))]
        if side == 's':
            indices = indices + [ (blk_idx, i, j) for i,j in
                        itertools.product(range(Nx), range(n)) ]
        if side == 'e':
            indices = indices + [ (blk_idx, i, j) for i,j in
                        itertools.product(range(Nx-2,Nx-n,-1), range(Ny)) ]
        if side == 'n':
            indices = indices + [ (blk_idx, i, j) for i,j in
                        itertools.product(range(Nx), range(Ny-2, Ny-n, -1)) ]

    return indices


def is_interactive():
    """ Check if Python is running in interactive mode. """
    import __main__ as main
    return not hasattr(main, '__file__')
