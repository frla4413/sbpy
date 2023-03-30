import numpy as np
from operators import SBP1D
import pdb

N = 11
x = np.linspace(0, 1, N)
dx = 1/(N-1)
sbp_op = SBP1D(N, dx, accuracy = 4)

foo = np.transpose(sbp_op.D).todense()
print(np.transpose(sbp_op.D)@x)

pdb.set_trace()

