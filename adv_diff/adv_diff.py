import numpy as np
import pdb
from tqdm import tqdm
from scipy import sparse
import warnings

def vec_to_tensor(grid, vec):
    shapes = grid.get_shapes()
    component_length = np.sum([Nx*Ny for (Nx,Ny) in shapes])

    vec = np.reshape(vec, (1, component_length))

    start = 0
    U = []
    for Nx, Ny in shapes:
        U.append(np.reshape(vec[0][start:(start+Nx*Ny)], (Nx,Ny)))
    return np.array(U)

def solution_to_file(grid, sol, name_base):
    for i in range(len(sol)):
        filename = name_base+str(i)+'.tec'
        export_to_tecplot(grid,sol[i],filename)

def export_to_tecplot(grid,sol,filename):
    blocks = grid.get_blocks()

    with open(filename,'w') as f:
        f.write('TITLE = "advection_solution.tec"\n')
        f.write('VARIABLES = "x","y","u"\n')

        for k in range(len(blocks)):
            X = blocks[k][0]
            Y = blocks[k][1]

            f.write('ZONE I = ' + str(X.shape[1])+ \
                    ', J = ' + str(X.shape[0])+ ', F = POINT\n')
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    my_str = str(X[i,j]) + ' ' + str(Y[i,j]) +\
                            ' ' + str(sol[i,j]) + '\n'
                    f.write(my_str)
        f.close()


def u_a(t,x,y):
    return np.sin(x-t) + np.sin(y-t)

def uy_a(t,x,y):
    return np.cos(y-t)

def force(t, x, y, a, b, e):
    ut_a = lambda t,x,y: -np.cos(x-t) - np.cos(y-t)
    ux_a = lambda t,x,y: np.cos(x -t)
    uxx_a = lambda t,x,y: -np.sin(x-t)
    uyy_a = lambda t,x,y: -np.sin(y-t)

    out =  ut_a(t,x,y) + a*ux_a(t,x,y) + b*uy_a(t,x,y) - \
           e*(uxx_a(t,x,y) + uyy_a(t,x,y))
    return out.flatten()

def force_time_independent(x, y, a, b, e):
    ux_a = lambda  x,y: np.cos(x)
    uxx_a = lambda x,y: -np.sin(x)
    uyy_a = lambda x,y: -np.sin(y)

    out =  a*ux_a(x,y) + b*uy_a(0,x,y) - \
           e*(uxx_a(x,y) + uyy_a(x,y))
    return out.flatten()


def spatial_op(t,u,sbp,a,b,e):
    u = vec_to_tensor(sbp.grid,u)
    ux = sbp.diffx(u)
    uy = sbp.diffy(u)
    uxx = sbp.diffx(ux)
    uyy = sbp.diffy(uy)
    L = a*ux + b*uy - e*(uxx + uyy)
    return L.flatten(), uy.flatten() 

def sat_op(t,u,uy,sbp,b,e,side):
    block_idx = 0
    u = vec_to_tensor(sbp.grid,u)
    uy = vec_to_tensor(sbp.grid,uy)

    bd_slice = sbp.grid.get_boundary_slice(block_idx, side)
    u_bd = u[block_idx][bd_slice]
    uy_bd = uy[block_idx][bd_slice]
    pinv = sbp.get_pinv(block_idx, side)
    bd_quad = sbp.get_boundary_quadrature(block_idx, side)
    lift = pinv*bd_quad

    N,M = sbp.grid.get_shapes()[0]
    num_blocks = sbp.grid.num_blocks
    s = np.zeros((num_blocks,N,M))

    blocks = sbp.grid.get_blocks()
    X = blocks[0][0]
    Y = blocks[0][1]
    x_bd = X[bd_slice]
    y_bd = Y[bd_slice]

    if side == 's':
        g = b*u_a(t,x_bd,y_bd) - e*uy_a(t,x_bd,y_bd)
        bc = b*u_bd - e*uy_bd - g
        s[block_idx][bd_slice] = lift*bc
    else:
        x_bd = X[bd_slice]
        y_bd = Y[bd_slice]

        g = uy_a(t,x_bd,y_bd)
        bc = e*(uy_bd - g)
        s[block_idx][bd_slice] = lift*bc
    return s.flatten()


def spatial_jacobian(sbp, a, b, e):
    
    block_idx = 0
    Dx = sbp.get_Dx(block_idx)
    Dy = sbp.get_Dy(block_idx)

    N,M = sbp.grid.get_shapes()[block_idx]
    return a*Dx + b*Dy - e*(Dx*Dx + Dy*Dy)

def sat_jacobian_south(sbp,a,b,e):

    side = 's'
    block_idx = 0
    bd_slice = sbp.grid.get_boundary_slice(block_idx, side)
    pinv = sbp.get_pinv(block_idx, side)
    bd_quad = sbp.get_boundary_quadrature(block_idx, side)
    lift = pinv*bd_quad
    Dy = sbp.get_Dy(block_idx)

    N,M = sbp.grid.get_shapes()[block_idx]
    Js = np.zeros((N,M))
    Js[bd_slice] = b*lift
    Js = sparse.diags(Js.flatten())
    
    if e != 0:
        lift_mat = np.zeros((N,M))
        lift_mat[bd_slice] = lift
        lift_mat = sparse.diags(lift_mat.flatten())
        Js += -e*lift_mat*Dy

    return Js

def sat_jacobian_north(sbp,a,b,e):

    block_idx = 0
    N,M = sbp.grid.get_shapes()[block_idx]
    if e  == 0:
        return sparse.csr_matrix((N*M,N*M))
    
    else:
        side = 'n'
        bd_slice = sbp.grid.get_boundary_slice(block_idx, side)
        pinv = sbp.get_pinv(block_idx, side)
        bd_quad = sbp.get_boundary_quadrature(block_idx, side)
        lift = pinv*bd_quad
        Dy = sbp.get_Dy(block_idx)

        lift_mat = np.zeros((N,M))
        lift_mat[bd_slice] = lift
        lift_mat = sparse.diags(lift_mat.flatten())
        return e*lift_mat*Dy
   
def solve(grid, spatial_operator, init, dt, num_steps, J_spatial):

    U=[]
    U.append(vec_to_tensor(grid,init))
    sol = init
    tol = 1e-11
    t   = 0
    for k in tqdm(range(num_steps)):
        t += dt
        if k == 0:
            prev_state = sol
            sol = backward_euler_step(spatial_operator, prev_state, dt, t, tol, J_spatial)
        else:
            prev_prev_state = prev_state
            prev_state = sol
            sol = bdf2_step(spatial_operator, prev_state, prev_prev_state, 
                            dt, t, tol, J_spatial)

    # only save last step
    sol = vec_to_tensor(grid, sol)
    U.append(sol)
    return U

def backward_euler_step(op, prev_state, dt, t, tol, J_spatial):
    N = int(len(prev_state))

    def F(new_state, prev_state):
        T = new_state - prev_state
        S = op(t,new_state)
        
        Jt = sparse.identity(N)
        return T+dt*S, Jt+dt*J_spatial

    new_state = prev_state.copy()
    n_iter = 0
    while True:
        L, J = F(new_state, prev_state)
        err = np.linalg.norm(L, ord=np.inf)

        if err < tol:
            break
        
        if err > 1e7 or n_iter > 50: #temp magic constant
            print("Error: " + str(err))
            print("Not solved")
            break

        delta = sparse.linalg.spsolve(J,L)
        new_state -= delta
        n_iter += 1

    return new_state

def bdf2_step(op, prev_state, prev_prev_state, dt, t, tol, J_spatial):
    N = int(len(prev_state))

    def F(new_state, prev_state):
        T = new_state - (4/3)*prev_state + (1/3)*prev_prev_state
        S = op(t,new_state)
        
        Jt = sparse.identity(N)
        return T+(2/3)*dt*S, Jt+(2/3)*dt*J_spatial

    new_state = prev_state.copy()
    n_iter = 0
    while True:
        L, J = F(new_state, prev_state)
        err = np.linalg.norm(L, ord=np.inf)

        if err < tol:
            break
        
        if err > 1e7 or n_iter > 50: #temp magic constant
            print("Error: " + str(err))
            print("Not solved")
            break

        delta = sparse.linalg.spsolve(J,L)
        new_state -= delta
        n_iter += 1
    return new_state

def solve_steady_state(grid, spatial_operator, init, J):

    sol = init
    tol = 1e-12
    max_iter = 10

    for k in range(max_iter):

        F = spatial_operator(sol)
        err = np.linalg.norm(F,ord=np.inf) 

        if err < tol:
            break

        print("error k: ", err)

        sol-= sparse.linalg.spsolve(J,F)

    sol = vec_to_tensor(grid, sol)

    print("Quit after", k, "iterations")
    print("Final residual: " + str(err) )
    return sol

