import numpy as np

from operators import SBP1D

N   = 20
x   = np.linspace(0,1,N)
dx  = x[1]-x[0]
one = np.ones(N)
acc = 8
sbp = SBP1D(N,1/(N-1),acc)
print(sbp.D.shape)

power = 4
f     = x**power
print(one@(sbp.P*f) -1/(power+1))

print(sbp.D*f-power*x**(power-1))
