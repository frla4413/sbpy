import numpy as np
import pdb
import scipy
import matplotlib.pyplot as plt
from matplotlib import rc,cm
from matplotlib.colors import LightSource
import sbpy.operators

'''
    bd_func__ are to be used in the zero-order model:
        nu_tau = bd_func__2 sqrt(2 Sij Sij)

    Use evaluate_1d_in_y_function_on_grid to get on a mesh grid.
'''

def bd_func_no_damping(x): 

    kappa = 0.41
    c = 10
    out = []
    ind = int(len(x)/2)
    
    for xi in x[0:ind]:
        yi = kappa*xi
        foo = np.amin([yi, c])
        out.append(foo)
    if len(x) % 2 == 0:
        out = np.array([out, np.flip(out)]).flatten()
    else:
        yi = kappa*x[ind]
        out.append(np.amin([yi,c]))
        out = np.concatenate([out, np.flip(out[0:-1])], axis = 0)
    return out

def bd_func_damping(x, e, u_tau, middle):

    kappa = 0.41
    out = []
    
    yi_func = lambda z: kappa*z*(1 - np.exp(-z*u_tau/e/26))
    for xi in x:
        if xi <= middle:
            yi = yi_func(xi)
        else:
            yi = yi_func(-xi+2*middle)
        out.append(yi)
    return out

def bd_func_damping_open(x, e, u_tau):

    kappa = 0.41
    yi_func = lambda z: kappa*z*(1 - np.exp(-z*u_tau/e/26))
    return yi_func(x)


#def bd_func_damping(x, e, u_tau):
#
#    kappa = 0.41
#    #c = 0.5
#    out = []
#    #ind = int(len(x)/2)
#    ind = np.argmax(x>=1)
#    for xi in x[0:ind]:
#        xip = xi*u_tau/e
#        yi = kappa*xi*(1 - np.exp(-xip/26))
#        #foo = yi#np.amin([yi, c])
#        out.append(yi)
#    if len(x) % 2 == 0:
#        out = np.array([out, np.flip(out)]).flatten()
#    else:
#        yi = kappa*xi*(1 - np.exp(-xip/26))
#        #yi = kappa*x[ind]*(1 - np.exp(-10*x[ind]))
#        #out.append(np.amin([yi, c]))
#        out.append(yi)
#        out = np.concatenate([out, np.flip(out[0:-1])], axis = 0)
#    return out


def bd_func_damping_open_top(x):

    kappa = 0.41
    c = 0.1
    out = []
    for xi in x:
        yi = kappa*xi*(1 - np.exp(-10*xi))
        foo = np.amin([yi, c])
        out.append(foo)
    return out

def evaluate_1d_in_y_function_on_grid(x, func_1d): 

    m = x.shape[0]
    out = np.zeros(x.shape)
    for i in range(m):
        out[i,:] = func_1d(x[i,:])
    return out

def determine_u_tau_from_low(u, y, uy_0 = 1, kappa = 0.41, C = 5, e = 0.01):
    f = lambda x: x*(np.log(y*x/e) + kappa*C)/kappa - u
    f_prime = lambda x: (np.log(y*x/e) + kappa*C + 1)/kappa
    u_tau =  scipy.optimize.fsolve(f, x0 = uy_0, fprime = f_prime)
    control = lambda x: (np.log(y*x/e) + kappa*C)/kappa - u/x

    print("root: " + str(control(u_tau)))
    return u_tau

def compute_beta(u_tau, kappa = 0.41, C = 5, e = 0.01):
    return e/u_tau*np.exp(-kappa*C)

def plot_plus_variables(sbp, U, u_tau, ax, title = 'u+', e = 0.01):
    blocks = sbp.grid.get_blocks()
    Y = blocks[0][1]
    slice1  = slice(1,None,None)

    x_ind  = 3
    y_plus = Y*u_tau/e
    u_plus = U[-1]/u_tau
    ax.semilogx(y_plus[x_ind,slice1], u_plus[x_ind,slice1], label=title,linewidth=3)
    return ax

def plot_solution_slice(sbp, U, ax, title = 'u'):
    blocks = sbp.grid.get_blocks()
    Y = blocks[0][1]
    slice1  = slice(0,None,None)

    x_ind  = 3
    ax.plot(Y[x_ind,slice1], U[-1][x_ind,slice1], label=title,linewidth=3)
    return ax


def plot_log_law(u_tau, ax, y0, y1, kappa = 0.41, C = 5, e = 0.01):

    log_law = lambda x: np.log(x)/kappa + C
    y_plus  = np.array([y0, y1])*u_tau/e

    ax.plot(y_plus,log_law(y_plus),'--')
    return ax

def read_ins_data(file_name):

    Y = []
    U = []
    for line in open(file_name):
        li=line.strip()
        if not li.startswith("#"):
            data = line.rstrip().split()
            Y.append(float(data[0]))
            U.append(float(data[1]))

    Y = np.array(Y)
    U = np.array(U)
    return Y,U

def read_full_ins_data(file_name):

    Y = []
    U = []
    for line in open(file_name):
        li=line.strip()
        if li.startswith("N"):
            data = line.rstrip().split()
            Nx = int(data[1])
            Ny = int(data[3])
        elif not li.startswith("#"):
            data = line.rstrip().split()
            Y.append(float(data[0]))
            U.append(float(data[1]))

    Y = np.array(Y)
    U = np.array(U)
    return Y.reshape(Nx,Ny), U.reshape(Nx,Ny)


def write_ins_data(file_name, y, u):

    f = open(file_name,"w")
    for i in range(len(y)):
        write_str = str(y[i]) + " " + str(u[i]) + "\n"
        f.write(write_str)
    f.close()

def write_full_ins_data(file_name, y, u):

    f = open(file_name,"w")
    (Nx, Ny) = u.shape
    y = y.flatten()
    u = u.flatten()
    write_str = str("Nx " + str(Nx) + " Ny " + str(Ny) + "\n")
    f.write(write_str)

    for i in range(len(y)):
        write_str = str(y[i]) + " " + str(u[i]) + "\n"
        f.write(write_str)
    f.close()

def get_plus_variables(y, u, u_tau, e):
    u_plus = u/u_tau 
    y_plus = y*u_tau/e
    return y_plus, u_plus

def compute_u_tau(y, u, e):
    Ny = len(u)
    dy = y[1] - y[0]
    sbp_op = sbpy.operators.SBP1D(Ny, dy)
    uy = sbp_op.D @ u

    uy = uy[0]
    tau_w = e*uy
    return np.sqrt(tau_w)

def get_solution_slice(sbp, U):
    u = U[-1][2]
    blocks = sbp.grid.get_blocks()
    Y = blocks[0][1]
    y = Y[1]
    uy = sbp.diffy([U[-1]])[0]
    return y, u, uy

def slice_to_full_grid_function(sbp, u_slice):
    ''' set a slice of a grid function to full grid '''
    blocks = sbp.grid.get_blocks()
    X = blocks[0][0]
    Nx, Ny = X.shape
    u_full = []
    if len(u_slice) == Ny:
        for k in range(Nx):
            u_full.append(u_slice)
    else:
        for k in range(Ny):
            u_full.append(u_slice)
    return np.array(u_full)


def read_ins_data_from_tecfile(file_name):

    X = []
    Y = []
    U = []
    V = []
    P = []
    for line in open(file_name):
        li=line.strip()
        if not li.startswith(("TITLE", "VARIABLES", "ZONE")):
            data = line.rstrip().split()
            X.append(float(data[0]))
            Y.append(float(data[1]))
            U.append(float(data[2]))
            V.append(float(data[3]))
            P.append(float(data[4]))

    X = np.array(X)
    Y = np.array(Y)
    U = np.array(U)
    V = np.array(V)
    P = np.array(P)
    return X, Y, U, V, P

def abl_surf_plot(X,Y,Z, title = None):

    fig,ax = plt.subplots(figsize=(11,11), subplot_kw={"projection":"3d"})
    #surf = ax.plot_surface(X, Y, Z, cmap=cm.jet,
    #               linewidth=0, antialiased=False, shade = True)
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    ls = LightSource(300, 20)
    rgb = ls.shade(Z, cmap=cm.jet, vert_exag=1, blend_mode='soft')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=rgb,
                               linewidth=0, antialiased=False, shade=False)
    plt.show()

def compute_low_error_robin(sbp, u, y0):

    blocks = sbp.grid.get_blocks()
    _,Y = blocks[0]
    uy = sbp.diffy([u])
    K = np.zeros(u.shape)
    K[:,1:] = Y[:,1:]*np.log(Y[:,1:]/y0)
    low_error = np.abs(u - K*uy)
    return low_error[0]

def compute_low_error_slice(y, u, uy, y0):

    K = y[1:]*np.log(y[1:]/y0)
    low_error = np.zeros(u.shape)
    low_error[1:] = np.abs(u[1:] - K*uy[1:])
    return low_error

def compute_uy_lw_error_slice(y ,u, a):

    Ny = len(u)
    dy = y[1] - y[0]
    sbp_op = sbpy.operators.SBP1D(Ny, dy)
    uy = sbp_op.D @ u
    gy = a/y[1:]
    err = np.zeros(u.shape)
    err[1:] = np.abs(uy[1:] - gy)
    return err

def post_process_fine_solution(file_name, e, plot = False, plot_plus_vars = False):
    ''' compute beta (= y0), u_tau 
        find the regression line u = a log y + b
        beta is determined  so that 
        u = a log y/beta 
    '''
    y, u = read_ins_data(file_name)
    Ny = len(u)
    dy = y[1] - y[0]
    sbp_op = sbpy.operators.SBP1D(Ny, dy)
    uy = sbp_op.D @ u
    tau_w = e * uy[0]
    u_tau = np.sqrt(tau_w)

    # u = a log y + b
    ind2 = 500
    ind1 = 400
    #ind2 = 265
    #ind1 = 240
    ind2 = 210
    ind1 = 205
    a = (u[ind2] - u[ind1])/(np.log(y[ind2]) - np.log(y[ind1]))
    b = u[ind2] - a*np.log(y[ind2])
    beta = np.exp(-b/a)
    kappa2 = a/u_tau
    C2 = b/u_tau - 1/kappa2*np.log(u_tau/e)
    pdb.set_trace()


    if plot:
        #log_line = lambda y: a*np.log(y) + b
        log_line = lambda y: a*np.log(y/beta)
        u_reg = log_line(y[ind1:ind2])
        print("err log_line: " + str(np.linalg.norm(u[ind1:ind2] - u_reg)))
        y_reg = np.array([y[40] , y[1800]])
        u_reg = log_line(y_reg)
        plt.rcParams.update({'font.size': 35})
        fig = plt.figure(figsize=(11,9))
        ax  = fig.gca()

        ind = int(len(u)/2) + 400
        ax.semilogx(y[0:ind], u[0:ind])
        #ax.plot(y, u)
        ax.semilogx(y_reg, u_reg, '--k')
        y_points = np.array([y[ind1], y[ind2]])
        u_points = log_line(y_points)
        ax.semilogx(y_points, u_points, 'xk')
        plt.title("Fine solution and regression line")
        plt.show()
        print("u_tau :"+ str(u_tau))
        print("beta :" + str(beta))

    if plot_plus_vars:
        log_line = lambda y_p: np.log(y_p)/0.33 + 4
        log_line = lambda yp: np.log(yp*e/u_tau/beta)*a/u_tau
        u_reg = log_line(y[ind1:ind2])
        y_reg = np.array([y[150] , y[1800]])
        u_reg = log_line(y_reg)
        ind = int(len(y))
        ind_mid = int(len(y)/2)
        y_plus = y*u_tau/e
        u_plus = u/u_tau
        plt.semilogx(y_plus[:ind], u_plus[:ind] - u_plus[0])

        y_vec = np.array([y_plus[45], y_plus[ind_mid]])
        plt.plot(y_plus[ind_mid], -10,'x')
        #y_vec = np.array([y[10]+1, y[ind]])
        u_vec = log_line(y_vec)
        plt.semilogx(y_vec, u_vec - u_plus[0],'--k')
        plt.show()
    return beta, u_tau, a, b

