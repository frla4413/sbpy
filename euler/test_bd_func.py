import pdb

import matplotlib.pyplot as plt
import numpy as np
import scipy
from sbpy.utils import surf_plot
from sbpy.abl_utils import bd_func_no_damping, bd_func_damping, evaluate_1d_in_y_function_on_grid, correct_bd_func_damping


def plot_1d_bd_func():
    x = np.linspace(0,2,100)
    y1 = bd_func_damping(x)
    plt.rcParams.update({'font.size': 35})
    fig = plt.figure(figsize=(11,9))
    ax = fig.gca()
    plt.title('l(y)')
    plt.plot(x,y1)
    plt.show()

def plot_2d_bd_func():

    e = 1/10000
    N = 100
    (X,Y) = np.meshgrid(np.linspace(0,1,5), np.linspace(0,2,N))
    X = np.transpose(X)
    Y = np.transpose(Y)
    bd_1d_func = lambda y: bd_func_damping(y,e)
    foo = evaluate_1d_in_y_function_on_grid(Y, bd_1d_func)
    surf_plot(X,Y,foo)

if __name__ == '__main__':
    plot_1d_bd_func()
