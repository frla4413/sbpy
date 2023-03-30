import numpy as np
import matplotlib.pyplot as plt
import pdb

def bd_func_damping(x, e, u_tau):

    if len(x) % 2 == 0:
        print("Must be odd number to get lm_func correct!")
        pdb.set_trace()
    kappa = 0.41
    out = []
    ind = np.argmax(x==1)
    pdb.set_trace()
    
    yi_func = lambda z: kappa*z*(1 - np.exp(-z*u_tau/e/26))
    for xi in x:
        if xi <= x[ind]:
            yi = yi_func(xi)
        else:
            yi = yi_func(-xi+2*x[ind])
        out.append(yi)
    return out

x = np.linspace(0,2,11)
e = 1e-4
u_tau = 0.044546384775524965

#x = np.linspace(0,1,25)
#c = 3
#b1 = x**c
#b2 = (1 - x)**c
#foo = b1/(b1 + b2)
#foo = 2*foo
#
#plt.plot(x, foo)
#
#x = np.linspace(0.1,1,25)
#c = 3
#b1 = x**c
#b2 = (1 - x)**c
#foo = b1/(b1 + b2)
#foo = 2*foo

y = bd_func_damping(x, u_tau,e)
plt.plot(x, x,'o')

plt.show()
