import numpy as np
import pdb
import matplotlib.pyplot as plt

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
    U = np.array(U).reshape(Nx,Ny)
    V = np.array(V).reshape(Nx,Ny)
    P = np.array(P).reshape(Nx,Ny)
    out_u = np.zeros((Nx-1,Ny))
    out_u[:int(Nx/2)-1] = U[:int(Nx/2)-1]
    out_u[int(Nx/2)-1] = 0.5*(U[int(Nx/2)-1] + U[int(Nx/2)])
    out_u[int(Nx/2):] = U[int(Nx/2)+1:]

    out_v = np.zeros((Nx-1,Ny))
    out_v[:int(Nx/2)-1] = V[:int(Nx/2)-1]
    out_v[int(Nx/2)-1] = 0.5*(V[int(Nx/2)-1] + V[int(Nx/2)])
    out_v[int(Nx/2):] = V[int(Nx/2)+1:]

    out_p = np.zeros((Nx-1,Ny))
    out_p[:int(Nx/2)-1] = P[:int(Nx/2)-1]
    out_p[int(Nx/2)-1] = 0.5*(P[int(Nx/2)-1] + P[int(Nx/2)])
    out_p[int(Nx/2):] = P[int(Nx/2)+1:]
    return out_u, out_v, out_p

Nx = 99
Ny = 39
diff = []
name_base = 'diff_plots/diff'
for k in range(700):
    U0,V0,P0 = load_u_interface(Nx+1,Ny,'small_domain/plots_interface_Nx100/sol'+str(k)+'.tec')
    X,Y,U1,V1,P1 = load_u_no_interface(Nx,Ny,'small_domain/plots_no_interface_Nx100/sol'+str(k)+'.tec')
    err = np.sqrt((U0-U1)**2 + (V0-V1)**2 +(P0-P1)**2)
    diff.append(np.linalg.norm(err,ord=np.inf))
    file_name = name_base + str(k)
#    export_to_tecplot(X,Y,err,file_name)
dt = 1e-2
t = np.linspace(0,dt*len(diff),len(diff))
plt.plot(t,diff)

Nx = 149
diff = []
for k in range(700):
    U0,V0,P0 = load_u_interface(Nx+1,Ny,'small_domain/plots_interface_Nx150/sol'+str(k)+'.tec')
    X,Y,U1,V1,P1 = load_u_no_interface(Nx,Ny,'small_domain/plots_no_interface_Nx150/sol'+str(k)+'.tec')
    err = np.sqrt((U0-U1)**2 + (V0-V1)**2 +(P0-P1)**2)
    diff.append(np.linalg.norm(err,ord=np.inf))
    file_name = name_base + str(k)
#    export_to_tecplot(X,Y,err,file_name)
plt.plot(t,diff)

Nx = 199
diff = []
name_base = 'diff_plots/diff'
for k in range(700):
    U0,V0,P0 = load_u_interface(Nx+1,Ny,'small_domain/plots_interface_Nx200/sol'+str(k)+'.tec')
    X,Y,U1,V1,P1 = load_u_no_interface(Nx,Ny,'small_domain/plots_no_interface_Nx200/sol'+str(k)+'.tec')
    err = np.sqrt((U0-U1)**2 + (V0-V1)**2 +(P0-P1)**2)
    diff.append(np.linalg.norm(err,ord=np.inf))
    file_name = name_base + str(k)
plt.plot(t,diff)
plt.xlabel("t")
plt.ylabel("err")
plt.show()
