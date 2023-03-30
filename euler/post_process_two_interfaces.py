import numpy as np
import pdb
import matplotlib.pyplot as plt
from sbpy.grid2d import MultiblockGrid, MultiblockSBP

def export_to_tecplot(X,Y,diff,filename):
    filename = filename +'.tec'
    with open(filename,'w') as f:
        f.write('TITLE = "diff_ine.tec"\n')
        f.write('VARIABLES = "x","y","diff"\n')

        f.write('ZONE I = ' + str(X.shape[1])+ \
                ', J = ' + str(X.shape[0])+ ', F = POINT\n')

        x = X.flatten()
        y = Y.flatten()
        diff = diff.flatten()
        for i in range(X.shape[1]*X.shape[0]):
                my_str = str(x[i]) + ' ' + str(y[i]) +' ' + str(diff[i]) +'\n'
                f.write(my_str)
        f.close()

def load_u_no_interface(Nx, Ny, filename):
    data = open(filename).read().strip().split("\n")
    X = []
    Y = []
    U = []
    V = []
    P = []
    for line in data:
        if line[0].isalpha() == False:
            split_line = line.split()
            X.append(float(split_line[0]))
            Y.append(float(split_line[1]))
            U.append(float(split_line[2]))
            V.append(float(split_line[3]))
            P.append(float(split_line[4]))
    X = np.array(X).reshape((Nx,Ny))
    Y = np.array(Y).reshape((Nx,Ny))
    U = np.array(U).reshape((Nx,Ny))
    V = np.array(V).reshape((Nx,Ny))
    P = np.array(P).reshape((Nx,Ny))
    return X,Y,U,V,P

def load_u_interface(Nx, Ny, filename):
    data = open(filename).read().strip().split("\n")
    U = []
    V = []
    P = []
    for line in data:
        if line[0].isalpha() == False:
            split_line = line.split()
            U.append(float(split_line[2]))
            V.append(float(split_line[3]))
            P.append(float(split_line[4]))
    U = np.array(U).reshape(3*Nx,Ny)
    V = np.array(V).reshape(3*Nx,Ny)
    P = np.array(P).reshape(3*Nx,Ny)
    out_u = np.zeros((3*Nx-2,Ny))
    out_u[:Nx-1] = U[:Nx-1]
    out_u[Nx-1] = 0.5*(U[Nx-1] + U[Nx])
    out_u[Nx:2*Nx-1] = U[Nx+1:2*Nx]
    out_u[2*Nx-1] = 0.5*(U[2*Nx-1] + U[2*Nx])
    out_u[2*Nx-1:] = U[2*Nx+1:]

    out_v = np.zeros((3*Nx-2,Ny))
    out_v[:Nx-1] = V[:Nx-1]
    out_v[Nx-1] = 0.5*(V[Nx-1] + V[Nx])
    out_v[Nx:2*Nx-1] = V[Nx+1:2*Nx]
    out_v[2*Nx-1] = 0.5*(V[2*Nx-1] + V[2*Nx])
    out_v[2*Nx-1:] = V[2*Nx+1:]

    out_p = np.zeros((3*Nx-2,Ny))
    out_p[:Nx-1] = P[:Nx-1]
    out_p[Nx-1] = 0.5*(P[Nx-1] + P[Nx])
    out_p[Nx:2*Nx-1] = P[Nx+1:2*Nx]
    out_p[2*Nx-1] = 0.5*(P[2*Nx-1] + P[2*Nx])
    out_p[2*Nx-1:] = P[2*Nx+1:]
    return out_u, out_v, out_p

def compute_diff_data(name_base_interface, name_base_no_interface, Nx):
    Ny = 39
    max_err = []
    l2_err = []

    blocks = []
    x = np.linspace(-1,1,2*Nx-1)
    y = np.linspace(0,1,Ny+1)
    [X0,Y0]  = np.meshgrid(x,y)
    X0 = np.transpose(X0)
    Y0 = np.transpose(Y0)
    X0 = X0[:,:-1]
    Y0 = Y0[:,:-1]
    blocks.append([X0,Y0])
    grid = MultiblockGrid(blocks)
    acc = 4
    sbp = MultiblockSBP(grid, accuracy_x = acc, accuracy_y=2,periodic=True)
    
    for k in range(700):
        U0,V0,P0 = load_u_interface(Nx,Ny,name_base_interface+str(k)+'.tec')
        X,Y,U1,V1,P1 = load_u_no_interface(3*Nx-2,Ny,name_base_no_interface+str(k)+'.tec')
        #err = ((U0-U1)**2 + (V0-V1)**2 +(P0-P1)**2)
        w0 = np.array([U0.flatten(), V0.flatten(),P0.flatten()]).flatten()
        w1 = np.array([U1.flatten(), V1.flatten(), P1.flatten()]).flatten()
        err = np.abs(w0-w1)
        max_err.append(np.max(err))
        err_tmp_l2 = 0
        #print(np.max([np.max(U0-U1), np.max(V0-V1)]), max_err)
        idx = 2*Nx-1
        err_tmp =  sbp.integrate([(U0[:idx] - U1[:idx])**2])
        err_tmp += sbp.integrate([(V0[:idx] - V1[:idx])**2])
        err_tmp += sbp.integrate([(P0[:idx] - P1[:idx])**2])
        l2_err.append(np.sqrt(err_tmp))
#        file_name = 'error_plots_discontinuous_Nx50/err' + str(k)
#        export_to_tecplot(X,Y,err,file_name)
    return max_err, l2_err



def generate_error_plot(diff):
    dt = 1e-2
    t = np.linspace(0,dt*len(diff),len(diff))
    plt.plot(t,diff)
    plt.xlabel("t")
    plt.ylabel("err")
    plt.show()

max_err,l2_err = compute_diff_data('interface_plots/discontinuous/Nx50/sol',
                   'interface_plots/no_interface/Nx50/sol',50)
dt = 1e-2
t = np.linspace(0,dt*len(max_err),len(max_err))

plt.plot(t[4:],max_err[4:])
plt.xlabel("t")
plt.ylabel("err")
plt.show()

