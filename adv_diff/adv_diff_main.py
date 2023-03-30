import pdb

import matplotlib.pyplot as plt
from scipy import integrate
import numpy as np
import scipy
from sbpy.grid2d import MultiblockGrid, MultiblockSBP
from sbpy.utils import (create_convergence_table, surf_plot)
from adv_diff import vec_to_tensor, force, u_a, spatial_op, solution_to_file,sat_op, solve, spatial_jacobian, sat_jacobian_south, sat_jacobian_north, solve_steady_state,force_time_independent

def adv_diff_test_explicit():

    a = 1.4
    b = 0.9
    e = 0.01
    tspan = [0, 1]

    N_vec = np.array([41,61])
    err = []

    for N in N_vec:
        (X,Y) = np.meshgrid(np.linspace(0,2*np.pi,N), np.linspace(0,2*np.pi,N))
        X = np.transpose(X)
        Y = np.transpose(Y)
        X = X[:-1]
        Y = Y[:-1]

        grid = MultiblockGrid([(X,Y)])
        sbp = MultiblockSBP(grid, accuracy = 2, periodic = True)
        (Nx,Ny) = X.shape

        init = u_a(0,X,Y).flatten()

        def adv_diff_spatial_op(t,u): 
            # ut + L(u) = f
            L,uy = spatial_op(t,u,sbp,a,b,e)
            s  = sat_op(t,u,uy,sbp,b,e,'s') +  \
                 sat_op(t,u,uy,sbp,b,e,'n')
            f  =  force(t, X, Y, a, b, e)
            return -L - s + f

        sol = integrate.solve_ivp(adv_diff_spatial_op,tspan,init, rtol=1e-12, atol=1e-12)
        t   = sol.t
        
        sol_out = []
        for k in range(len(t)):
            sol_out.append(sol.y[:,k].reshape(Nx,Ny))
        final_sol = sol_out[-1]
        err_u = u_a(tspan[-1],X,Y) - final_sol
        err.append(np.sqrt(sbp.integrate([err_u*err_u])))

    create_convergence_table(N_vec, err, 1/(N_vec-1))
    return grid,sol_out

def adv_diff_test_implicit():

    a = 1.4
    b = 0.8
    e = 5
    tspan = [0, 1]

    dt = 0.01
    num_steps = int(np.ceil((tspan[1] - tspan[0])/dt))

    tspan[1] = tspan[0] + dt*num_steps

    N_vec = np.array([61,81])
    err = []

    for N in N_vec:
        (X,Y) = np.meshgrid(np.linspace(0,2*np.pi,N), np.linspace(0,2*np.pi,N))
        X = np.transpose(X)
        Y = np.transpose(Y)
        X = X[:-1]
        Y = Y[:-1]

        grid = MultiblockGrid([(X,Y)])
        sbp = MultiblockSBP(grid, accuracy = 2, periodic = True)
        
        init = u_a(0,X,Y).flatten()

        def adv_diff_spatial_op(t,u):
            # ut + L(u) + S(u) = f
            L,uy = spatial_op(t,u,sbp,a,b,e)
            s  = sat_op(t,u,uy,sbp,b,e,'s') +  \
                 sat_op(t,u,uy,sbp,b,e,'n')
            f  =  force(t, X, Y, a, b, e)
            return L + s - f
        J_spatial = spatial_jacobian(sbp, a, b, e) + \
                    sat_jacobian_south(sbp,a,b,e) + \
                    sat_jacobian_north(sbp,a,b,e)

        sol = solve(grid, adv_diff_spatial_op, init, dt, num_steps,J_spatial)
        err_u = u_a(tspan[1],X,Y) - sol[-1]
        err.append(np.sqrt(sbp.integrate([err_u*err_u])))

    create_convergence_table(N_vec, err, 1/(N_vec-1))
    return grid,sol

def adv_diff_test_steady_state():

    a = 1.4
    b = 0.5
    e = 5

    N_vec = np.array([61,81,161])
    err = []

    for N in N_vec:
        (X,Y) = np.meshgrid(np.linspace(0,2*np.pi,N), np.linspace(0,2*np.pi,N))
        X = np.transpose(X)
        Y = np.transpose(Y)
        X = X[:-1]
        Y = Y[:-1]

        grid = MultiblockGrid([(X,Y)])
        sbp = MultiblockSBP(grid, accuracy = 2, periodic = True)
        
        init = np.ones((N-1)*N)

        def adv_diff_spatial_op(u):
            t = 0
            # L(u) + S(u) = f
            L,uy = spatial_op(t,u,sbp,a,b,e)
            s  = sat_op(t,u,uy,sbp,b,e,'s') +  \
                 sat_op(t,u,uy,sbp,b,e,'n')
            f  =  force_time_independent(X, Y, a, b, e)
            return L + s - f

        J_spatial = spatial_jacobian(sbp, a, b, e) + \
                    sat_jacobian_south(sbp,a,b,e) + \
                    sat_jacobian_north(sbp,a,b,e)

        sol = solve_steady_state(grid, adv_diff_spatial_op, init, J_spatial)
        err_u = u_a(0,X,Y) - sol[-1]
        err.append(np.sqrt(sbp.integrate([err_u*err_u])))

    create_convergence_table(N_vec, err, 1/(N_vec-1))
    return grid,sol


if __name__ == '__main__':

#    grid,sol = adv_diff_test_explicit()
#    grid,sol = adv_diff_test_implicit()
    grid,sol = adv_diff_test_steady_state()
    #solution_to_file(grid, sol, 'plots/movie')
    #grid.plot_grid_function(sol[-1])
