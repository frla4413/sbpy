import pdb
import warnings

import numpy as np
import scipy
from scipy import sparse
from tqdm import tqdm
from sbpy import operators
from sbpy.dg_operators import legendre_gauss_lobatto_nodes_and_weights,polynomial_derivative_matrix

def solve_ivp(spatial_operator, init, dt, num_steps, incompressible = False):
    """ Solves an initial-value problem.
    Arguments:
        spatial_operator: A spatial operator, including the SATs
        init:             Initial data 
        incompressible:   True = no time-dependence on last variable
                          False= time-dependence on all variables

    Returns:
        sol_out: A list of multiblock functions representing the solution vector 
                 in each time step.
    """
    sol_out = []
    sol     = init
    tol     = 1e-10
    sol_out.append(sol)

    for k in tqdm(range(num_steps)):
        sol = bdf1_step(k*dt,spatial_operator, sol, dt, tol, incompressible)
        sol_out.append(sol)
    return sol_out

def bdf1_step(t, op, prev_state, dt, tol, incompressible):
    N = int(len(prev_state)/3)

    def F(new_state):
        
        if incompressible:
            I = sparse.identity(N)
            O = sparse.csr_matrix((N,N))

            T = np.concatenate(
              [(new_state[0:int(2*N)] - prev_state[0:int(2*N)]), np.zeros(N)])

            #Jacobian
            Jt = sparse.bmat([[I, O, O],
                              [O, I, O],
                              [O, O, O]])
        else: 
            T = np.concatenate([(new_state - prev_state)])
            #Jacobian
            Jt = sparse.identity((N*N,N*N))

        S,Js = op(t+dt,new_state)
        return T+dt*S, Jt+dt*Js

    new_state = prev_state.copy()
    n_iter    = 0
#    delta = scipy.optimize.newton_krylov(F,new_state,verbose=True,
#                                         maxiter=150,f_tol = tol)
#    L = F(delta)
#    err = np.linalg.norm(L, ord=np.inf)
#    print(err)
#    new_state = delta
    while True:
        L, J = F(new_state)
        err = np.linalg.norm(L, ord=np.inf)
#        print(err)
        
        if err < tol:
            break

        if err > 1e5 or n_iter > 10: #temp magic constant
            print("Error :" + str(err))
            print("Iterations :" + str(n_iter))
            raise Exception("No solution found. Try decreasing dt.")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                delta = sparse.linalg.spsolve(J,L)
        except sparse.linalg.MatrixRankWarning:
            #Small perturbation if singular Jacobian
            delta = np.random.normal(scale=1e-7, size=len(prev_state))

        new_state -= delta
        n_iter += 1
    return new_state


def bdf2_step(t, op, prev_prev_state, prev_state, dt, tol, incompressible):
    N = int(len(prev_state)/3)

    def F(new_state, prev_prev_state, prev_state):
        
        if incompressible:
            I = sparse.identity(N)
            O = sparse.csr_matrix((N,N))

            T = np.concatenate(
              [(new_state[0:int(2*N)] - (4/3)*prev_state[0:int(2*N)]
                +(1/3)*prev_prev_state[0:int(2*N)]), np.zeros(N)])

            #Jacobian
            Jt = sparse.bmat([[I, O, O],
                              [O, I, O],
                              [O, O, O]])
        else: 
            T = np.concatenate([(new_state - prev_state)])
            #Jacobian
            Jt = sparse.identity((N*N,N*N))

        S,Js = op(t+dt,new_state)
        return T+(2/3)*dt*S, Jt+(2/3)*dt*Js

    new_state = prev_state.copy()
    n_iter    = 0
    while True:
        L, J = F(new_state, prev_prev_state, prev_state)
        err = np.linalg.norm(L, ord=np.inf)
        
        if err < tol:
            break

        if err > 1e5 or n_iter > 10: #temp magic constant
            print("Error :" + str(err))
            print("Iterations :" + str(n_iter))
            raise Exception("No solution found. Try decreasing dt.")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                delta = sparse.linalg.spsolve(J,L)
        except sparse.linalg.MatrixRankWarning:
            #Small perturbation if singular Jacobian
            delta = np.random.normal(scale=1e-7, size=len(prev_state))

        new_state -= delta
        n_iter += 1

    return new_state


def sbp_in_time_step_dg(t, op, cur_state, dt, tol):

    num_nodes     = 3
    nodes,weights = legendre_gauss_lobatto_nodes_and_weights(num_nodes-1)
    nodes         = nodes*0.5*dt + t + 0.5*dt # adjust to interval [t, t + dt]
    weights       = weights*dt
    D             = polynomial_derivative_matrix(nodes)
    P             = sparse.diags([weights], [0])
    N             = int(len(cur_state)/3)
    I_til         = sparse.csr_matrix([[1,0,0],[0,1,0],[0,0,0]])
    I             = sparse.identity(N)
    I_til         = sparse.kron(I_til, I)
    D_t           = sparse.kron(D, sparse.kron(np.identity(3),I))
    D_big_tilde   = sparse.kron(np.identity(num_nodes), I_til)@D_t
    E0            = np.zeros((num_nodes,num_nodes))
    E0[0,0]       = 1/weights[0]
    Pen           = sparse.kron(E0,I_til)
    Jt            = D_big_tilde + Pen # constant part of Jacobian

    def F(state):
        T        = D_big_tilde@state
        T[0:2*N]+= (1/weights[0])*(state[0:2*N]-cur_state[0:2*N]) # SAT-time
        S = []
        J = []
        for i in range(num_nodes):
            Si,Ji = op(nodes[i],state[i*3*N:(i+1)*3*N])
            S = np.concatenate([S, Si])
            J.append(Ji)
        #Jacobian
        J = Jt + sparse.block_diag(J)
        return T+S, J

    state = []
    for i in range(num_nodes):
        state = np.concatenate([state,cur_state.copy()])

    n_step = 0
    while True:
        L, J = F(state)
        err  = np.linalg.norm(L, ord=np.inf)
        #print(err)
    
        if err < tol:
            break

        if n_step > 10:
            print("No solution in sbp_in_time_step_dg at time " + str(t) + "\n")
            print("error: " + str(err) + "\n")
            break

        state  -=sparse.linalg.spsolve(J,L)
        n_step +=1

    #print("Error: {:.2e}".format(err))
    return state[-3*N:] # new_state last in state-vector


def solve_steady_state(spatial_operator, init):
    """ Solves the steady-state version of an Euler problem.
    Arguments:
        spatial_operator: A spatial operator built from the euler_operator 
            plus SAT operators.
        init: Initial guess compatible with spatial operator

    Returns:
        sol_out: A list of multiblock functions representing the solution vector 
                 in each time step.

    """
    sol_array = []
    sol       = init
    max_iter  = 60
    tol       = 1e-12
    alpha     = 0.5
    sol_array.append(sol)
    
    for k in range(max_iter):

        F,J = spatial_operator(sol)
        err = np.linalg.norm(F,ord=np.inf) 

        if err < 0.1:
            alpha = 1

        if err < tol:
            break

        print("error: ", err)

        sol-= alpha*sparse.linalg.spsolve(J,F)
        sol_array.append(sol)

    print(np.linalg.norm(F,ord=np.inf)) 

    print("Converged in ", k, "iterations")
    #err = []

    # print errors for convergence rate in Newton steps
    #for k in range(len(U)):
    #    err.append(np.linalg.norm(U[-1]-U[k],ord=np.inf))

    #print("Error in inf-norm, u^* is the last iterate")
    #print(err)
    return sol_array
