import pdb

import matplotlib.pyplot as plt
import numpy as np
import scipy
from sbpy.grid2d import MultiblockGrid, MultiblockSBP
from sbpy.utils import (create_convergence_table, export_to_tecplot, surf_plot,
                       get_gauss_initial_data)
from sbpy.abl_utils import read_ins_data

#import mms_kovaszany as mms
import mms as mms
import time


from euler import (euler_operator, force_operator, inflow_operator,
                   outflow_operator, pressure_operator, wall_operator,
                   solve, solve_with_mpt, solve_steady_state, 
                   solve_steady_state_newton_krylov,
                   interior_penalty_robin_operator,
                   interior_penalty_dirichlet_operator)

def write_error_data(file_name, t, error):

    f = open(file_name,"w")
    for i in range(len(t)):
        write_str = str(t[i]) + " " + str(error[i]) + "\n"
        f.write(write_str)
    f.close()

def plot_mpt_data():
    t, err_no_mpt = read_ins_data("data_files/time_dependent_mms/no_mpt.dat")
    t, err_one_mpt = read_ins_data("data_files/time_dependent_mms/many_mpt_1.dat")
    t, err_three_mpt = read_ins_data("data_files/time_dependent_mms/many_mpt_3.dat")
    t, err_five_mpt = read_ins_data("data_files/time_dependent_mms/many_mpt_5.dat")
#    t, err_one_mpt = read_ins_data("data_files/time_dependent_mms/one_mpt.dat")
#    t, err_three_mpt = read_ins_data("data_files/time_dependent_mms/three_mpt.dat")
#    t, err_five_mpt = read_ins_data("data_files/time_dependent_mms/five_mpt.dat")
    slice1 = slice(0,len(t))
    linewidth = 3

    plt.rcParams["figure.figsize"] = (10,8)
    fig = plt.figure()
    plt.rcParams["text.usetex"] = True
    font = {'size' : 25}
    plt.rc('font', **font)
    plt.plot(t[slice1], err_no_mpt[slice1], 'k', label ="No penalty terms", linewidth=linewidth)
    plt.plot(t[slice1], err_one_mpt[slice1], ':b', label="One penalty term", linewidth=linewidth)
    plt.plot(t[slice1], err_three_mpt[slice1],'-.r', label="Three penalty terms", linewidth=linewidth)
    plt.plot(t[slice1], err_five_mpt[slice1], '--g', label="Five penalty terms", linewidth=linewidth)
#    plt.plot(t[slice1], err_many_mpt[slice1], '-xg', label="Five penalties", linewidth=linewidth)
    plt.xticks([0,5,10])
    #plt.yticks([0,0.02, 0.04])
    plt.yticks([0,0.005, 0.01])
    plt.xlim(0,t[-1])
    plt.ylim(0,0.01)
    plt.legend()
    #plt.subplots_adjust(bottom=0, top=0.95, left=0, right=1)
    plt.xlabel("t")
    #plt.title("No penalties")
    plt.ylabel("$\|e\|$")
    path = "/home/fredrik/work/abl/abl_report/img/mms_figures/"
    fig_name = "mms_N11_sbp21_time_error_u_not_remove_mpt_zoom.png"
#    fig.savefig(path + fig_name)
    plt.show()

def time_dependent_mms(acc = 2, mpt = False):
    # need import mms_file as mms on top  to run
    # remember: epsilon in mms-file and here need to agree!

    e = mms.e
    N = 11
    (X,Y) = np.meshgrid(np.linspace(0,1,N), np.linspace(1,2,N))
    X     = np.transpose(X)
    Y     = np.transpose(Y)

    grid = MultiblockGrid([(X,Y)])
    sbp  = MultiblockSBP(grid, accuracy_x = acc, accuracy_y = acc)

    initu = mms.u(0,X,Y)
    initv = mms.v(0,X,Y)
    initp = mms.p(0,X,Y)

    data = np.zeros(X.shape)
    beta = mms.beta
    def spatial_op(state, t):
        S,J,ux,uy,vx,vy = euler_operator(sbp, state, e)

        Sbd,Jbd = inflow_operator(sbp,state,0,'w',\
                            mms.wn_data(sbp,0,'w', t), \
                            mms.wt_data(sbp,0,'w',t), e) + \
                  wall_operator(sbp,state,0,'s', e) + \
                  pressure_operator(sbp, state, 0, 'e', ux, uy, vx, vy,\
                            mms.normal_outflow_data(sbp,0,'e',t), \
                            mms.tangential_outflow_data(sbp,0,'e',t), e) + \
                  pressure_operator(sbp, state, 0, 'n', ux, uy, vx, vy,\
                            mms.normal_outflow_data(sbp,0,'n',t), \
                            mms.tangential_outflow_data(sbp,0,'n',t), e) 
        S     -= force_operator(sbp,mms.force1,mms.force2,mms.force3,t)[0]
        return S+Sbd, J+Jbd

    dt = 1/100
    num_timesteps = 1000
    start = time.time()
    if mpt:
        U,V,P, time_error,time_to_solve = solve_with_mpt(grid, spatial_op, initu, initv,initp, dt, num_timesteps, sbp, mpt = True)
    else:
        U,V,P, time_error,time_to_solve = solve_with_mpt(grid, spatial_op, initu, initv,initp, dt, num_timesteps, sbp, mpt = False)
    print("exe time: " + str(time.time() - start))
    print(np.mean(time_to_solve))

    t_end = dt*num_timesteps
    err_u   = np.abs(U[-1] - mms.u(t_end,X,Y))
    err_v   = np.abs(V[-1] - mms.v(t_end,X,Y))
    err_p   = np.abs(P[-1] - mms.p(t_end,X,Y))
    err_tot = np.sqrt(sbp.integrate([err_u*err_u]) +
                      sbp.integrate([err_v*err_v]) + 
                      sbp.integrate([err_p*err_p]))

    error_u = np.sqrt(sbp.integrate([err_u*err_u]))
    #low_error = compute_low_error_robin(sbp, U[-1], beta)
    #low_error = err_u
    #ind_max = np.unravel_index(np.argmax(low_error, axis=None), low_error.shape)
    #print(error_u)
    #print(ind_max)
    #print(low_error[slice1,slice2])
    #print(low_error[ind_max])
    #grid.plot_grid_function(U[-1])
    #pdb.set_trace()
    t_vec = np.linspace(0,t_end, num_timesteps)
    #write_error_data('data_files/time_dependent_mms/many_mpt_5.dat',t_vec, time_error)
    plt.plot(t_vec,time_error)
    plt.show()
    #grid.plot_grid_function(U[-1])
    return grid,U,V,P

if __name__ == '__main__':
    grid,U,V,P = time_dependent_mms(acc = 2, mpt=True)
    #export_to_tecplot(grid,U[-1],V[-1],P[-1],"test_mpt.tec")
    #plot_mpt_data()
    

