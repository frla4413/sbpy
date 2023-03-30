""" This module contains classes for solving partial differential equations on
multiblock grids. """

import sys
import numpy as np
from scipy import integrate
from scipy.stats import norm

from sbpy import operators
from sbpy import grid2d
from sbpy import utils

if utils.is_interactive():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


_SIDES = ['s', 'e', 'n', 'w']


class AdvectionDiffusionSolver:
    """ A multiblock linear scalar advection-diffusion solver.

    Based on the interface coupling in Carpenter & Nordström (1998)"""

    def __init__(self, grid, **kwargs):
        """ Initializes an AdectionDiffusionSolver object.

        Arguments:
            grid: A MultiblockSBP object.
        Optional:
            initial_data: A multiblock function containing initial data.
            source_term: A function F(t,x,y) representing the source term.
            velocity: A pair [a,b] specifying the flow velocity.
            diffusion: The diffusion coefficient.
            internal_data: A list of values to be enforced internally (used
                together with the internal_indices argument).
            internal_indices: A list of indices of the form (blk, i, j) used
                together with internal_data to enforce internal data.
            u: The exact solution (used in run_mms_test()).
            ut: The t-derivative of the exact solution.
            ux: The x-derivative of the exact solution.
            uy: The y-derivative of the exact solution.
            uxx: The second x-derivative of the exact solution.
            uyy: The second y-derivative of the exact solution.
        """

        self.grid = grid
        self.t = 0
        self.U   = np.array([ np.zeros(shape) for shape in grid.get_shapes() ])
        self.Ux  = np.array([ np.zeros(shape) for shape in grid.get_shapes() ])
        self.Uy  = np.array([ np.zeros(shape) for shape in grid.get_shapes() ])
        self.Uxx = np.array([ np.zeros(shape) for shape in grid.get_shapes() ])
        self.Uyy = np.array([ np.zeros(shape) for shape in grid.get_shapes() ])
        self.Ut  = np.array([ np.zeros(shape) for shape in grid.get_shapes() ])
        self.mms = False

        if 'diffusion' in kwargs:
            self.eps = kwargs['diffusion']
        else:
            self.eps = 0.01

        if 'initial_data' in kwargs:
            assert(grid.is_shape_consistent(kwargs['initial_data']))
            self.U = kwargs['initial_data']

        if 'source_term' in kwargs:
            self.source_term = kwargs['source_term']
        else:
            self.source_term = None

        if 'velocity' in kwargs:
            self.velocity = kwargs['velocity']
        else:
            self.velocity = np.array([1.0,-1.0])/np.sqrt(2)

        if 'internal_data' in kwargs:
            assert('internal_indices' in kwargs)
            self.internal_data = kwargs['internal_data']
        else:
            self.internal_data = None

        if 'internal_indices' in kwargs:
            assert('internal_data' in kwargs)
            self.internal_indices = kwargs['internal_indices']
        else:
            self.internal_indices = None

        if 'u' in kwargs:
            self.u = kwargs['u']
        if 'ut' in kwargs:
            self.ut = kwargs['ut']
        if 'ux' in kwargs:
            self.ux = kwargs['ux']
        if 'uxx' in kwargs:
            self.uxx = kwargs['uxx']
        if 'uy' in kwargs:
            self.uy = kwargs['uy']
        if 'uyy' in kwargs:
            self.uyy = kwargs['uyy']


        # Save bool arrays determining inflows. For example, if inflow[k]['w'][j]
        # is True, then the j:th node of the western boundary of the k:th block
        # is an inflow node.
        self.inflow = [ {} for _ in range(self.grid.num_blocks) ]
        for k in range(self.grid.num_blocks):
            for side in ['s', 'e', 'n', 'w']:
                bd = self.grid.get_normals(k, side)
                inflow = np.array([ self.velocity@n < 0 for n in bd ],
                                  dtype = bool)
                self.inflow[k][side] = inflow

        # Compute flow velocity at boundaries
        self.flow_velocity = [ {} for _ in range(self.grid.num_blocks) ]
        for block_idx in range(self.grid.num_blocks):
            for side in _SIDES:
                normals = grid.get_normals(block_idx, side)
                self.flow_velocity[block_idx][side] = \
                    np.array([ self.velocity@n for n in normals ])

        # Compute interface alphas
        self.alphas = [ {} for _ in range(self.grid.num_blocks) ]
        for k in range(self.grid.num_blocks):
            for side in _SIDES:
                if grid.is_interface(k,side):
                    self.alphas[k][side] = self._compute_alpha(k, side)

        # Save penalty coefficients for each interface
        self.inviscid_if_coeffs = [ {} for _ in range(self.grid.num_blocks) ]
        self.viscid_if_coeffs = [ {} for _ in range(self.grid.num_blocks) ]
        for (idx1, side1), (idx2, side2) in self.grid.get_interfaces():
            normals    = grid.get_normals(idx1, side1)
            flow_vel   = np.array([ self.velocity@n for n in normals ])

            pinv1 = self.grid.sbp_ops[idx1].pinv[side1]
            bdquad1 = self.grid.sbp_ops[idx1].boundary_quadratures[side1]
            pinv2 = self.grid.sbp_ops[idx2].pinv[side2]
            bdquad2 = self.grid.sbp_ops[idx2].boundary_quadratures[side2]

            alpha1 = self.alphas[idx1][side1]
            alpha2 = self.alphas[idx2][side2]

            s1_viscid = -1.0
            s2_viscid = 0.0
            s1_inviscid = 0.5*flow_vel - 0.25*self.eps*\
                    (s1_viscid**2/alpha1 + s2_viscid**2/alpha2)
            s2_inviscid = s1_inviscid - flow_vel

            self.inviscid_if_coeffs[idx1][side1] = \
                s1_inviscid*pinv1*bdquad1
            self.inviscid_if_coeffs[idx2][side2] = \
                s2_inviscid*pinv2*bdquad2
            self.viscid_if_coeffs[idx1][side1] = \
                s1_viscid*self.eps*pinv1*bdquad1
            self.viscid_if_coeffs[idx2][side2] = \
                s2_viscid*self.eps*pinv2*bdquad2

        # Set boundary types
        for bd_idx in range(grid.num_boundaries):
            if grid.get_boundary_info(bd_idx) is None:
                grid.set_boundary_info(bd_idx,
                        {'type': 'char',
                         'inflow_data': lambda t,x,y: 0,
                         'outflow_data': lambda t,x,y,: 0})


    def _compute_alpha(self, block_idx, side):
        vol_quad = self.grid.sbp_ops[block_idx].volume_quadrature
        vol_quad = grid2d.get_function_boundary(vol_quad, side)
        normals = self.grid.get_normals(block_idx, side)
        nx = normals[:,0]
        ny = normals[:,1]
        bd_quad = self.grid.sbp_ops[block_idx].boundary_quadratures[side]
        alpha1 = vol_quad/(nx**2 * bd_quad+1e-14)
        alpha2 = vol_quad/(ny**2 * bd_quad+1e-14)
        alphas = np.array([ min(a1, a2) for a1,a2 in zip(alpha1, alpha2) ])
        return 0.5*alphas


    def _update_sol(self, U):
        self.U = U


    def _compute_spatial_derivatives(self):
        self.Ux = self.grid.diffx(self.U)
        self.Uy = self.grid.diffy(self.U)
        self.Uxx = self.grid.diffx(self.Ux)
        self.Uyy = self.grid.diffy(self.Uy)


    def _differential_operator(self):
        a = self.velocity[0]
        b = self.velocity[1]
        return -a*self.Ux-b*self.Uy+self.eps*(self.Uxx+self.Uyy)


    def _compute_temporal_derivative(self):
        a = self.velocity[0]
        b = self.velocity[1]
        self.Ut = self._differential_operator()

        if self.mms:
            source_term = [self.ut(self.t,X,Y) +
                           a*self.ux(self.t,X,Y) +
                           b*self.uy(self.t,X,Y) +
                           -self.eps*(self.uxx(self.t,X,Y) + self.uyy(self.t,X,Y)) for
                           X,Y in self.grid.get_blocks()]
        elif self.source_term is not None:
            source_term = [self.source_term(self.t,X,Y) for
                           X,Y in self.grid.get_blocks() ]
        else:
            source_term = self.grid.num_blocks*[0]

        for (k,F) in enumerate(source_term):
            self.Ut[k] += F


        # Add interface penalties
        for (if_idx,interface) in enumerate(self.grid.get_interfaces()):
            ((idx1,side1),(idx2,side2)) = interface
            bd_slice1 = self.grid.get_boundary_slice(idx1, side1)
            bd_slice2 = self.grid.get_boundary_slice(idx2, side2)
            u  = self.U[idx1][bd_slice1]
            ux = self.Ux[idx1][bd_slice1]
            uy = self.Uy[idx1][bd_slice1]
            v  = self.U[idx2][bd_slice2]
            vx = self.Ux[idx2][bd_slice2]
            vy = self.Uy[idx2][bd_slice2]
            normals = self.grid.get_normals(idx1, side1)
            fluxu = normals[:,0]*ux + normals[:,1]*uy
            fluxv = normals[:,0]*vx + normals[:,1]*vy

            s1_invisc = self.inviscid_if_coeffs[idx1][side1]
            s1_visc   = self.viscid_if_coeffs[idx1][side1]
            s2_invisc = self.inviscid_if_coeffs[idx2][side2]
            s2_visc   = self.viscid_if_coeffs[idx2][side2]


            if self.grid.is_flipped_interface(if_idx):
                self.Ut[idx1][bd_slice1] += s1_invisc*(u-np.flip(v))
                self.Ut[idx1][bd_slice1] += s1_visc*(fluxu - np.flip(fluxv))
                self.Ut[idx2][bd_slice2] += s2_invisc*(v-np.flip(u))
                self.Ut[idx2][bd_slice2] += s2_visc*(fluxv - np.flip(fluxu))
            else:
                self.Ut[idx1][bd_slice1] += s1_invisc*(u-v)
                self.Ut[idx1][bd_slice1] += s1_visc*(fluxu - fluxv)
                self.Ut[idx2][bd_slice2] += s2_invisc*(v-u)
                self.Ut[idx2][bd_slice2] += s2_visc*(fluxv - fluxu)


        # Add external boundary penalties
        for (bd_idx, (block_idx, side)) in enumerate(self.grid.get_boundaries()):
            bd_info  = self.grid.get_boundary_info(bd_idx)
            bd_slice = self.grid.get_boundary_slice(block_idx, side)
            pinv     = self.grid.sbp_ops[block_idx].pinv[side]
            bd_quad  = self.grid.sbp_ops[block_idx].boundary_quadratures[side]
            inflow   = self.inflow[block_idx][side]
            outflow  = np.invert(inflow)
            sigma    = pinv*bd_quad
            u        = self.U[block_idx][bd_slice]
            ux       = self.Ux[block_idx][bd_slice]
            uy       = self.Uy[block_idx][bd_slice]
            normals  = self.grid.get_normals(block_idx, side)
            flow_vel = self.flow_velocity[block_idx][side]
            flux     = normals[:,0]*ux + normals[:,1]*uy
            (x,y)    = self.grid.get_boundary(block_idx,side)
            (X,Y)    = self.grid.get_block(block_idx)

            if self.mms:
                u_exact     = self.u(self.t,x,y)
                ux_exact    = self.ux(self.t,x,y)
                uxx_exact   = self.uxx(self.t,x,y)
                uy_exact    = self.uy(self.t,x,y)
                uyy_exact   = self.uyy(self.t,x,y)
                flux_exact = np.array([ux*n1 + uy*n2 for (ux,uy,(n1,n2)) in
                                       zip(ux_exact,uy_exact,normals)])

            if bd_info['type'] == 'dirichlet':
                if self.mms:
                    data = u_exact
                else:
                    g = bd_info['data']
                    data = g(self.t,x,y)

                self.Ut[block_idx][bd_slice] += -sigma*(u-data)

            if bd_info['type'] == 'char':
                g_in  = bd_info['inflow_data']
                g_out = bd_info['outflow_data']
                if self.mms:
                    inflow_data = flow_vel*u_exact - self.eps*flux_exact
                    outflow_data = self.eps*flux_exact
                else:
                    inflow_data = g_in(self.t,x,y)
                    outflow_data = g_out(self.t,x,y)
                in_bc    = flow_vel*u - self.eps*flux
                out_bc   = self.eps*flux
                in_diff = in_bc - inflow_data
                out_diff = out_bc - outflow_data
                self.Ut[block_idx][bd_slice] += sigma*inflow*in_diff
                self.Ut[block_idx][bd_slice] += -sigma*outflow*out_diff

        # Add internal penalties
        if self.internal_data is not None:
            for (k,(blk, i, j)) in enumerate(self.internal_indices):
                self.Ut[blk][i,j] -= 100*(self.U[blk][i,j] - self.internal_data[k])



    def set_boundary_condition(self, boundary_index, condition):
        """ Set the boundary condition for a given boundary.

        You can see the indices of external boundaries by running
        plot_domain(boundary_indices=True) on your grid.

        Arguments:
            boundary_index: The index of the boundary you wish to set a boundary
                condition for.

            condition: To specify a dirichlet condition, supply a dict of the
                form {'type': 'dirichlet', 'data': g}, where g = g(t,x,y).
                To specify characteristic conditions (i.e. a Robin condition at
                inflow nodes and a Neumann condition at outflow nodes), supply a
                dict of the form
                {'type': 'char', 'inflow_data': g, 'outflow_data': h}, where
                g = g(t,x,y) and h=h(t,x,y).
        """
        assert('type' in condition)
        if ('type' == 'dirichlet'):
            assert('data' in condition)
        if ('type' == 'char'):
            assert('inflow_data' in condition)
            assert('outflow_data' in condition)
        self.grid.set_boundary_info(boundary_index, condition)


    def solve(self, tspan):
        init = grid2d.multiblock_to_array(self.grid, self.U)

        pbar = tqdm(total=100, leave=False)
        def f(t, y):
            pbar.n = int((100*(t-tspan[0])/(tspan[1]-tspan[0])))
            pbar.update()
            U = grid2d.array_to_multiblock(self.grid, y)
            self.t = t
            self._update_sol(U)
            self._compute_spatial_derivatives()
            self._compute_temporal_derivative()

            return np.concatenate([ ut.flatten() for ut in self.Ut ])

        eval_pts = np.linspace(tspan[0], tspan[1], int(30*(tspan[1]-tspan[0])))
        self.sol = integrate.solve_ivp(f, tspan, init,
                                       rtol=1e-10, atol=1e-10,
                                       t_eval=eval_pts)
        pbar.close()


    def run_mms_test(self, tspan):
        """ Check simulation against an exact solution.

        Note that this method requires that the object was initialized with the
        exact solution and its derivatives.

        Arguments:
            tspan: A pair of starting and ending times.

        Returns:
            err: The L2-error at the final time.
        """


        self.mms = True
        assert(hasattr(self, 'u'))
        assert(hasattr(self, 'ux'))
        assert(hasattr(self, 'uxx'))
        assert(hasattr(self, 'uy'))
        assert(hasattr(self, 'uyy'))

        errs = []

        self.U = [ self.u(tspan[0], X, Y) for X,Y in self.grid.get_blocks() ]

        self.solve(tspan)

        final_time = self.sol.t[-1]
        U = []
        for frame in np.transpose(self.sol.y):
            U.append(grid2d.array_to_multiblock(self.grid, frame))

        U_exact = self.grid.evaluate_function(lambda x,y: self.u(final_time, x, y))

        err = [ (u - u_exact)**2 for (u,u_exact) in zip(U[-1],U_exact) ]
        err = np.sqrt(self.grid.integrate(err))

        self.mms = False

        return err

