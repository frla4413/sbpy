""" This module contains various utility functions. """

import pdb
import itertools
import numpy as np
from sbpy import grid2d

def get_gauss_initial_data(X, Y, cx, cy):
    """
    Use to produce initial data formed as gaussian pulse in u
    v,p are constant zeros

    Parameter: 
        X,Y:    grid points
        cx,cy:  center points of the pulse
    Output: 
        u,v,p

    """
    rv1 = multivariate_normal([cx,cy], 0.01*np.eye(2))
    gauss_bell = rv1.pdf(np.dstack((X,Y)))
    normalize = rv1.pdf(np.dstack((cx,cy)))
    initu = 2*np.array([gauss_bell])/normalize
    initv = np.array([np.zeros(X.shape)])
    initp = np.array([np.ones(X.shape)])

    return initu, initv, initp


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


def solution_to_file(grid,U,V,P,name_base):

    for i in range(len(U)):
        filename = name_base+str(i)
        export_to_tecplot(grid,U[i],V[i],P[i],filename)

def export_to_tecplot(grid,U,V,P,filename):
    blocks = grid.get_blocks()

    filename = filename +'.tec'
    with open(filename,'w') as f:
        f.write('TITLE = "incompressible_navier_stokes_solution.tec"\n')
        f.write('VARIABLES = "x","y","u","v","p"\n')

        for k in range(len(blocks)):
            X = blocks[k][0]
            Y = blocks[k][1]
            f.write('ZONE I = ' + str(X.shape[1])+ \
                    ', J = ' + str(X.shape[0])+ ', F = POINT\n')

            for j in range(X.shape[1]):
                for i in range(X.shape[0]):
                    my_str = str(X[i,j]) + ' ' + str(Y[i,j]) +\
                            ' ' + str(U[k][i,j]) + ' ' + str(V[k][i,j]) +\
                            ' ' + str(P[k][i,j]) + '\n'
                    f.write(my_str)
        f.close()
