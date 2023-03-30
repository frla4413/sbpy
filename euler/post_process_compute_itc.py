import numpy as np
import pdb
import matplotlib.pyplot as plt
from sbpy.grid2d import MultiblockGrid, MultiblockSBP

def load_w_interface(Nx, Ny, filename):
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
    U = np.array(U).reshape(3,Nx,Ny)
    V = np.array(V).reshape(3,Nx,Ny)
    P = np.array(P).reshape(3,Nx,Ny)
    return U,V,P

def compute_itc(base_name, Nx):
    Ny = 39
    blocks = []
    blocks = []
    x = np.linspace(-1,0,Nx)
    y = np.linspace(0,1,Ny+1)
    [X0,Y0]  = np.meshgrid(x,y)
    X0 = np.transpose(X0)
    Y0 = np.transpose(Y0)
    X0 = X0[:,:-1]
    Y0 = Y0[:,:-1]
    blocks.append([X0,Y0])

    x = np.linspace(0,1,Nx)
    [X1,Y1]  = np.meshgrid(x,y)
    X1 = np.transpose(X1)
    Y1 = np.transpose(Y1)
    X1 = X1[:,:-1]
    Y1 = Y1[:,:-1]
    blocks.append([X1,Y1])

    x = np.linspace(1,2,Nx)
    [X2,Y2]  = np.meshgrid(x,y)
    X2 = np.transpose(X2)
    Y2 = np.transpose(Y2)
    X2 = X2[:,:-1]
    Y2 = Y2[:,:-1]
    blocks.append([X2,Y2])
    grid = MultiblockGrid(blocks)

    acc = 4
    sbp = MultiblockSBP(grid, accuracy_x = acc, accuracy_y=2,
                        periodic=True)

    itc_max = []
    for k in range(70):
        file_name = base_name + str(k) + '.tec'
        U,V,P = load_w_interface(Nx,Ny,file_name)

        idx_L = 0
        bd_slice_L= sbp.grid.get_boundary_slice(0, 'e')
        u_L       = U[idx_L][bd_slice_L]
        v_L       = V[idx_L][bd_slice_L]
        p_L       = P[idx_L][bd_slice_L]

        idx_R = 1
        bd_slice_R= sbp.grid.get_boundary_slice(1, 'w')
        u_R       = U[idx_R][bd_slice_R]
        v_R       = V[idx_R][bd_slice_R]
        p_R       = P[idx_R][bd_slice_R]

        bd_quad   = sbp.get_boundary_quadrature(idx_L, 'w')

        u_bar = 0.5*(u_L + u_R)
        v_bar = 0.5*(v_L + v_R)
        u_j = u_L - u_R
        v_j = v_L - v_R
        p_j = p_L - p_R

        wL = np.array([u_L,v_L,p_L]).reshape(3,Ny)
        wR = np.array([u_R,v_R,p_R]).reshape(3,Ny)

        #(A_bar + E_bar)*w_jump
        A_E_wj = np.array([[2*u_bar*u_j + p_j],[v_bar * u_j + u_bar*v_j],[u_j]]).reshape(3,Ny)
        # (-C_L + C_R)*w_jump
        A_E_2_wj = np.array([[3*u_bar*u_j/2 + p_j],[v_bar*u_j/2 + u_bar*v_j],
                    [u_j]]).reshape(3,Ny)

        CL_DL_wj3 = u_j/2
        CR_DR_wj3 = -u_j/2

        term2 = np.array([[CL_DL_wj3],[CL_DL_wj3],[np.zeros(u_j.shape)]]).reshape(3,Ny)

        term3 = np.array([[CR_DR_wj3],[CR_DR_wj3],[np.zeros(u_j.shape)]]).reshape(3,Ny)

        itc = A_E_wj - A_E_2_wj - 0.5*term2*wL + 0.5*term3*wR
        itc_max.append(np.linalg.norm(itc,ord=np.inf))

    return itc_max


itc_max = compute_itc("long_domain/plots_interface/Nx75/sol",75)

t = np.linspace(0,0.7,70)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 30}

plt.rc('font',**font)
plt.plot(t[0:-1:10],itc_max[0:-1:10])
plt.legend(['Nx XX'])
plt.xlabel('t')
plt.ylabel('err')
plt.xticks([0, 0.3, 0.6])
#plt.yticks([0,5e-16,1e-15])
plt.show()
