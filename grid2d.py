""" This module contains functions and classes for managing 2D grids. The
conceptual framework used throughout the module is that 2D numpy arrays represent
function evaluations associated to some grid. For example, if f is an Nx-by-Ny
numpy array, then f[i,j] is interpreted as the evaluation of some function f in
an associated grid node (x_i, y_j). 2D numpy arrays representing function
evaluations on a grid are called 'grid functions'. We refer to the boundaries of
a grid function as 's' for south, 'e' for east, 'n' for north, and 'w' for west.
More precisely the boundaries of a grid function f are

    South: f[:,0]
    East:  f[-1,:]
    North: f[:,-1]
    West:  f[0,:]

Grids (also referred to as blocks) are stored as pairs of matrices (X,Y), such
that (X[i,j], Y[i,j]) is the (i,j):th node in the grid. Multiblock grids can be
thought of as lists of such pairs. A list F of grid functions is called a
'multiblock function' and should be interpreted as function evaluations on a
sequence of grids constituting a multiblock grid.
"""

from enum import Enum
import itertools
import pdb

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc,cm
from scipy import sparse

rc('text', usetex=True)

from sbpy import operators


_SIDES = ['s', 'e', 'n', 'w']

def flatten_multiblock_vector(vec):
    """Concatenates a gridfunction. Must be used if the blocks are of different shapes."""
    return np.concatenate([ u.flatten() for u in vec])

def allocate_gridfunction(grid):
    """Allocate a gridfunction on a multiblockgrid with different shapes on the blocks."""
    out = []
    for shape in grid.get_shapes():
        out.append(np.zeros(shape))
    return out

def collocate_corners(blocks, tol=1e-15):
    """ Collocate corners of blocks if they are equal up to some tolerance. """
    for ((X1,Y1),(X2,Y2)) in itertools.combinations(blocks,2):
        for (c1,c2) in itertools.product([(0,0),(-1,0),(-1,-1),(0,-1)], repeat=2):
            if np.abs(X1[c1]-X2[c2]) < tol and np.abs(Y1[c1]-Y2[c2]) < tol:
                X1[c1] = X2[c2]
                Y1[c1] = Y2[c2]

def get_boundary(X,Y,side):
    """ Returns the boundary of a block. """

    assert(side in {'w','e','s','n'})

    if side == 'w':
        return X[0,:], Y[0,:]
    elif side == 'e':
        return X[-1,:], Y[-1,:]
    elif side == 's':
        return X[:,0], Y[:,0]
    elif side == 'n':
        return X[:,-1], Y[:,-1]


def get_function_boundary(F,side):
    """ Returns the boundary of a grid function. """

    assert(side in {'w','e','s','n'})

    if side == 'w':
        return F[0,:]
    elif side == 'e':
        return F[-1,:]
    elif side == 's':
        return F[:,0]
    elif side == 'n':
        return F[:,-1]

def get_corners(X,Y):
    """ Returns the corners of a block.

    Starts with (X[0,0], Y[0,0]) and continues counter-clockwise.
    """
    return np.array([[X[0,0]  , Y[0,0]  ],
                     [X[-1,0] , Y[-1,0] ],
                     [X[-1,-1], Y[-1,-1]],
                     [X[0,-1] , Y[0,-1]]])

def get_center(X,Y):
    """ Returns the center point of a block. """
    corners = get_corners(X,Y)
    return 0.25*(corners[0] + corners[1] + corners[2] + corners[3])


def array_to_multiblock(grid, array):
    """ Converts a flat array to a multiblock function. """
    shapes = grid.get_shapes()
    F = [ np.zeros(shape) for shape in shapes ]

    counter = 0
    for (k,(Nx,Ny)) in enumerate(shapes):
        F[k] = np.reshape(array[counter:(counter+Nx*Ny)], (Nx, Ny))
        counter += Nx*Ny

    return F


def multiblock_to_array(grid, F):
    """ Converts a multiblock function to a flat array. """
    return np.array(F).flatten()


class MultiblockGrid:
    """ Represents a structured multiblock grid.

    Attributes:
        num_blocks: The total number of blocks in the grid.
    """

    def __init__(self, blocks):
        """ Initializes a Multiblock object.

        Args:
            blocks: A list of pairs of 2D numpy arrays containing x- and y-values
                   for each block.

            Note that the structure of these blocks should be such that for the
            k:th element (X,Y) in the blocks list, we have that (X[i,j],Y[i,j])
            is the (i,j):th node in the k:th block.
        """

        for (X,Y) in blocks:
            assert(X.shape == Y.shape)

        self.blocks = blocks
        self.num_blocks = len(blocks)

        self.shapes = []
        for (X,Y) in blocks:
            self.shapes.append((X.shape[0], X.shape[1]))

        # Save unique corners
        self.corners = []
        for X,Y in self.blocks:
            self.corners.append(get_corners(X,Y))

        self.corners = np.unique(np.concatenate(self.corners), axis=0)

        # Save faces in terms of unique corners
        self.faces = []

        for k,(X,Y) in enumerate(self.blocks):
            block_corners = get_corners(X,Y)
            indices = []
            for c in block_corners:
                idx = np.argwhere(np.all(c == self.corners, axis=1)).item()
                indices.append(idx)
            self.faces.append(np.array(indices))
        self.faces = np.array(self.faces)

        # Save unique edges
        self.edges = []
        for face in self.faces:
            for k in range(4):
                self.edges.append(np.array(sorted([face[k], face[(k+1)%4]])))

        self.edges = np.unique(self.edges, axis=0)

        # Save face edges
        self.face_edges = []
        for face in self.faces:
            self.face_edges.append({})
            for k,side in enumerate(_SIDES):
                edge = np.array(sorted([face[k], face[(k+1)%4]]))
                self.face_edges[-1][side] = \
                    np.argwhere(np.all(edge == self.edges, axis=1)).item()

        # Find interfaces
        self.block_interfaces = [{} for _ in range(self.num_blocks)]
        for ((i,edges1), (j,edges2)) in \
        itertools.combinations(enumerate(self.face_edges),2):
            for (side1,side2) in \
            itertools.product(_SIDES, repeat=2):
                if edges1[side1] == edges2[side2]:
                    self.block_interfaces[i][side1] = (j, side2)
                    self.block_interfaces[j][side2] = (i, side1)

        self.interfaces = []
        for k in range(len(self.edges)):
            blocks = []
            sides = []
            for n in range(self.num_blocks):
                for side in _SIDES:
                    if self.face_edges[n][side] == k:
                        blocks.append(n)
                        sides.append(side)
            if len(blocks) == 2:
                self.interfaces.append(((blocks[0],sides[0]),(blocks[1],sides[1])))

        # Find external boundaries
        self.boundaries = []
        for block_idx in range(self.num_blocks):
            for side in _SIDES:
                if not self.is_interface(block_idx, side):
                    self.boundaries.append((block_idx, side))

        self.num_boundaries = len(self.boundaries)
        self.boundary_info = [ None for _ in self.boundaries ]

        # Save boundary slices
        self.bd_slice_dicts = \
                [{'s': (slice(Nx), 0),
                  'e': (-1, slice(Ny)),
                  'n': (slice(Nx), -1),
                  'w': (0, slice(Ny))} for (Nx,Ny) in self.shapes]

    def evaluate_function(self, f):
        """ Evaluates a (vectorized) function on the grid. """
        return [ f(X,Y) for (X,Y) in self.blocks ]


    def get_blocks(self):
        """ Returns a list of matrix pairs (X,Y) representing grid blocks. """
        return self.blocks


    def get_block(self, k):
        """ Returns a matrix pair (X,Y) representing the k:th block. """
        return self.blocks[k]

    def get_X(self, k):
        """ Returns a matrix pair (X,Y) representing the k:th block. """
        return self.blocks[k][0]

    def get_Y(self, k):
        """ Returns a matrix pair (X,Y) representing the k:th block. """
        return self.blocks[k][1]


    def is_shape_consistent(self, F):
        """ Check if a multiblock function F is shape consistent with grid. """
        is_consistent = True
        for (k,f) in enumerate(F):
            if F[k].shape != self.shapes[k]:
                is_consistent = False
        return is_consistent


    def get_boundary_slice(self,k,side):
        """ Get a slice representing the boundary of block. The slice can
        be used to index the given boundary of a grid function on the given block.
        For example, if slice = get_boundary_slice(k,'w') and F is a grid function,
        then F[slice] will refer to the western boundary of F.

        Args:
            k: A block index.
            side: The side at which the boundary is located ('s', 'e', 'n', or 'w')

        Returns:
            slice: A slice that can be used to index the given boundary in F.
        """
        return self.bd_slice_dicts[k][side]


    def get_interfaces(self):
        """ Returns a list of pairs of the form ( (k1, s1), (k2, s2) ), where
            k1, k2 are the indices of the blocks connected to the interface, and
            s1, s2 are the sides of the respective blocks that make up the
            interface. """
        return self.interfaces


    def get_block_interfaces(self):
        """ Returns a list of dictionaries containing the interfaces for each
        block. For example, if interfaces = get_block_interfaces(), and
        interfaces[i] = {'n': (j, 'w')}, then the northern boundary of the block
        i coincides with the western boundary of the western boundary of block j.
        """
        return self.block_interfaces


    def get_boundary(self, block_idx, side):
        """ Returns a pair (x,y) of numpy arrays representing the boundary nodes
        of the specified boundary.

        Arguments:
            block_idx: The index of the block.
            side: The boundary side ('w','e','s', or 'n')
        """
        bd_slice = self.get_boundary_slice(block_idx, side)
        X,Y = self.blocks[block_idx]
        return (X[bd_slice],Y[bd_slice])


    def get_boundaries(self):
        """ Returns a list of pairs defining the external boundaries of the
        domain. For example, if bds = get_boundaries(), then each element of
        ext_bds is a pair of the form (k, side), where side = 'w', 'e' 's', or
        'n', specifying a block and its side constituting an external boundary.
        """
        return self.boundaries
    
    def bd_func_to_grid_func(self, F, block_idx, side):
        """ Returns the grid function with placed at boundary side """
        assert(side in {'w','e','s','n'})
        #print("test")

        X,Y = self.blocks[block_idx]
        if side == 'w':
            row_pos = np.arange(X.shape[0])
            col_pos = np.zeros(X.shape[1])
            return sparse.csr_matrix((F, (row_pos, col_pos)), shape=X.shape)
        elif side == 'e':
            row_pos= np.arange(X.shape[1])
            col_pos = (X.shape[1]-1)*np.ones(X.shape[0])
            return sparse.csr_matrix((F, (row_pos, col_pos)), shape=X.shape)
        elif side == 's':
            row_pos = np.zeros(X.shape[0])
            col_pos = np.arange(X.shape[1])
            #col_pos = np.arange(X.shape[0])
            #row_pos = np.arange(X.shape[0])
            #col_pos = np.zeros(X.shape[0])
            return sparse.csr_matrix((F, (row_pos, col_pos)), shape=X.shape)
        elif side == 'n':
            row_pos = (X.shape[0]-1)*np.ones(X.shape[0])
            col_pos = np.arange(row_pos.shape[0])
            #col_pos = (X.shape[0])*np.ones(X.shape[0])
            #row_pos = np.arange(col_pos.shape[0])

            #col_pos = (X.shape[0]-1)*np.ones(X.shape[0])

            return sparse.csr_matrix((F, (row_pos, col_pos)), shape=X.shape)

    
    def get_shapes(self):
        """ Returns a list of the shapes of the blocks in the grid. """
        return self.shapes


    def is_interface(self, block_idx, side):
        """ Check if a given side is an interface.

        Returns True if the given side of the given block is an interface. """

        if side in self.block_interfaces[block_idx]:
            return True
        else:
            return False

    def is_flipped_interface(self, interface_idx):
        """ Check if an interface has flipped orientation compared to its
        neighbor, such as, for example, an east-to-south interface.

        Arguments:
            interface_idx: The index of the interface.

        Returns:
            True if flipped, False otherwise.
        """
        is_flipped = False
        ((_,side1),(_,side2)) = self.interfaces[interface_idx]
        if (side1, side2) in [('s','e'), ('s','s'),
                              ('e','s'), ('e','e'),
                              ('n','w'), ('n','n'),
                              ('w','n'), ('w','w')]:
            is_flipped = True

        return is_flipped


    def plot_grid(self):
        """ Plot the entire grid. """

        fig, ax = plt.subplots()
        for X,Y in self.blocks:
            ax.plot(X,Y,'b')
            ax.plot(np.transpose(X),np.transpose(Y),'b')
            for side in {'w', 'e', 's', 'n'}:
                x,y = get_boundary(X,Y,side)
                ax.plot(x,y,'k',linewidth=3)
                ax.text(np.mean(x),np.mean(y),side)

        ax.axis('equal')
        ax.set_xlim([0,1])
        plt.show()


    def plot_domain(self, **kwargs):
        """ Fancy domain plot without gridlines.

        Arguments:
            boundary_indices: True or False. Draws indices at the boundaries.
            interface_indices: True or False. Draws indices at the interfaces.
        """

        interface_indices = False

        if 'boundary_indices' in kwargs:
            boundary_indices = kwargs['boundary_indices']
        else:
            boundary_indices = False

        if 'interface_indices' in kwargs:
            interface_indices = kwargs['interface_indices']
        else:
            interface_indices = False


        fig, ax = plt.subplots()
        for k,(X,Y) in enumerate(self.blocks):
            xs,ys = get_boundary(X,Y,'s')
            xe,ye = get_boundary(X,Y,'e')
            xn,yn = get_boundary(X,Y,'n')
            xn = np.flip(xn)
            yn = np.flip(yn)
            xw,yw = get_boundary(X,Y,'w')
            xw = np.flip(xw)
            yw = np.flip(yw)
            x_poly = np.concatenate([xs,xe,xn,xw])
            y_poly = np.concatenate([ys,ye,yn,yw])

            ax.fill(x_poly,y_poly,'tab:gray')
            ax.plot(x_poly,y_poly,'k')
            c = get_center(X,Y)
            ax.text(c[0], c[1], "$\Omega_" + str(k) + "$", fontsize=20,
                    fontweight='bold')

        # Draw boundary indices
        if boundary_indices:
            for (bd_idx, (block_idx, side)) in enumerate(self.boundaries):
                X,Y = self.blocks[block_idx]
                xb,yb = get_boundary(X,Y,side)
                xc = np.median(xb)
                yc = np.median(yb)
                ax.text(xc, yc, str(bd_idx), fontsize=20,
                        fontweight='bold')

        # Draw interface indices
        if interface_indices:
            for (if_idx, interface) in enumerate(self.interfaces):
                (blk_idx,side),(_,_) = interface
                X,Y = self.blocks[blk_idx]
                xb,yb = get_boundary(X,Y,side)
                xc = np.median(xb)
                yc = np.median(yb)
                ax.text(xc, yc, str(if_idx), fontsize=20,
                        fontweight='bold')

        ax.axis('equal')
        plt.show()

    def plot_grid_function(self, F, title = None):
        ''' Plot a grid function on a single block (block 0) '''

        #fig = plt.figure(figsize=(11,11))
        #ax = fig.gca(projection='3d')
        X,Y = self.get_blocks()[0]
        fig,ax = plt.subplots(figsize=(11,11), subplot_kw={"projection":"3d"})
        fig = ax.plot_surface(X, Y, F, cmap=cm.jet,
                   linewidth=0, antialiased=False)

        if title is not None:
            ax.set_title(title)
        plt.show()


    def get_neighbor_boundary(self, F, block_idx, side):
        """ Returns an array of boundary data from a neighboring block.

        Arguments:
            F: A 2d array of function evaluations on the neighbor block.
            block_idx: The index of the block to send data to.
            side: The side of the block to send data to ('s', 'e', 'n', or 'w').
        """
        assert(self.is_interface(block_idx, side))

        neighbor_idx, neighbor_side = self.block_interfaces[block_idx][side]

        flip = False
        if (neighbor_side, side) in [('s','e'), ('s','s'),
                                     ('e','s'), ('e','e'),
                                     ('n','w'), ('n','n'),
                                     ('w','n'), ('w','w')]:
            flip = True

        if flip:
            return np.flip(get_function_boundary(F, neighbor_side))
        else:
            return get_function_boundary(F, neighbor_side)


    def set_boundary_info(self, boundary_index, info):
        """ Adds user-defined information to a boundary.  Typically used to
        specify the boundary condition to be used.

        Arguments:
            boundary_index: The index of the boundary in the list returned by
                get_boundaries(). See plot_domain(boundary_indices=True)

            info: The information to assign to the boundary (for example a dict).
        """
        self.boundary_info[boundary_index] = info

    def get_boundary_info(self, boundary_index):
        """ Returns user-defined information associated to a boundary. """
        return self.boundary_info[boundary_index]


class MultiblockSBP:
    """ A class combining MultiblockGrid functionality and SBP2D functionality.  """

    def __init__(self, grid, accuracy_x = 2, accuracy_y = 2, periodic = False):
        """ Initializes a MultiblockSBP object.
        Args:
            grid: A MultiblockGrid object.
        Optional:
            accuracy: The interior accuracy of the difference operators (2 or 4).
        """

        self.grid = grid

        # Create SBP2D objects for each block.
        self.sbp_ops = []
        for (X,Y) in self.grid.get_blocks():
            if periodic:
                self.sbp_ops.append(operators.SBP2DPeriodic(X,Y,accuracy_x,accuracy_y))
            else:
                self.sbp_ops.append(operators.SBP2D(X,Y,accuracy_x, accuracy_y))

    def diffx(self, U):
        """ Differentiates a Multiblock function with respect to x. """
        return np.array([ sbp.diffx(u) for
                          sbp,u in zip(self.sbp_ops, U) ])


    def diffy(self, U):
        """ Differentiates a Multiblock function with respect to y. """
        return np.array([ sbp.diffy(u) for
                          sbp,u in zip(self.sbp_ops, U) ])


    def integrate(self, U):
        """ Integrates a Multiblock function over the domain. """
        return sum([ sbp.integrate(u) for
                     sbp,u in zip(self.sbp_ops, U) ])


    def get_normals(self, block_idx, side):
        """ Get the normals of a specified side of a particular block. """
        return self.sbp_ops[block_idx].normals[side]


    def get_pinv(self, block_idx, side):
        """ Get the inverse of the volume quadrature at a specified side
        of a particular block. """
        return self.sbp_ops[block_idx].pinv[side]

    def get_boundary_quadrature(self, block_idx, side):
        """ Get the boundary quadrature at a specified side of a particular
        block. """
        return self.sbp_ops[block_idx].boundary_quadratures[side]

    def get_full_P(self, block_idx):
        """ Get the boundary quadrature of a particular block. """
        return self.sbp_ops[block_idx].P

    def get_full_pinv(self, block_idx):
        """ Get the boundary quadrature of a particular block. """
        return self.sbp_ops[block_idx].Pinv


    def get_sbp_ops(self):
        """ Returns a list of SBP2D objects associated to each block. """
        return self.sbp_ops


    def get_Dx(self, block_idx):
        """ Get Dx for a given block. """
        return self.sbp_ops[block_idx].Dx


    def get_Dy(self, block_idx):
        """ Get Dy for a given block. """
        return self.sbp_ops[block_idx].Dy



#class MultiblockGridSBP(MultiblockGrid):
#    """ A class combining MultiblockGrid functionality and SBP2D functionality.  """
#
#    def __init__(self, blocks, accuracy = 2):
#        """ Initializes a MultiblockSBP object.
#        Args:
#            blocks: A list of matrix pairs representing the blocks.
#        Optional:
#            accuracy: The interior accuracy of the difference operators (2 or 4).
#        """
#        super().__init__(blocks)
#
#        # Create SBP2D objects for each block.
#        self.sbp_ops = []
#        for (X,Y) in self.get_blocks():
#            self.sbp_ops.append(operators.SBP2D(X,Y,accuracy))
#
#
#    def diffx(self, U):
#        """ Differentiates a Multiblock function with respect to x. """
#        return np.array([ self.sbp_ops[i].diffx(U[i]) for
#                          i in range(self.num_blocks) ])
#
#
#    def diffy(self, U):
#        """ Differentiates a Multiblock function with respect to y. """
#        return np.array([ self.sbp_ops[i].diffy(U[i]) for
#                          i in range(self.num_blocks) ])
#
#    def integrate(self, U):
#        """ Integrates a Multiblock function over the domain. """
#        return sum([ self.sbp_ops[i].integrate(U[i]) for
#                     i in range(self.num_blocks) ])
#
#    def get_normals(self, block_idx, side):
#        """ Get the normals of a specified side of a particular block. """
#        return self.sbp_ops[block_idx].normals[side]
#
#
#    def get_sbp_ops(self):
#        """ Returns a list of SBP2D objects associated to each block. """
#        return self.sbp_ops


def load_p3d(filename):
    with open(filename) as data:
        num_blocks = int(data.readline())

        X = []
        Y = []
        Nx = []
        Ny = []
        for _ in range(num_blocks):
            size = np.fromstring(data.readline(), sep=' ', dtype=int)
            Nx.append(size[0])
            Ny.append(size[1])

        blocks = []
        for k in range(num_blocks):
            X_cur = []
            Y_cur = []
            for n in range(Nx[k]):
                X_cur.append(np.fromstring(data.readline(), sep=' '))
            for n in range(Nx[k]):
                Y_cur.append(np.fromstring(data.readline(), sep=' '))

            blocks.append((np.array(X_cur),np.array(Y_cur)))
            #X.append(np.array(X_cur))
            #Y.append(np.array(Y_cur))
            for _ in range(Nx[k]):
                next(data)


    return blocks
