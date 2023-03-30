import pdb
import numpy as np
import matplotlib.pyplot as plt
from sbpy.abl_utils import read_ins_data, get_plus_variables

# plot dns data
u_tau = 0.05
e = 1/10000
file_name = "um.dat"
y_dns, u_dns = read_ins_data(file_name)

fig_dns = plt.figure()
ax_dns = fig_dns.gca()
ax_dns.plot(y_dns,u_dns)

ind = int(len(y_dns)/2)
y_dns_p, u_dns_p = get_plus_variables(y_dns[0:ind], u_dns[0:ind], u_tau, e)

file_name = "um_plus.dat"
y_dns_plus, u_dns_plus = read_ins_data(file_name)

# compare um and umean in log-scale --> verifies how to compute u^+ and y^+ and u_tau
fig_dns_log_plus = plt.figure()
ax_dns_log_plus = fig_dns_log_plus.gca()
ax_dns_log_plus.semilogx(y_dns_plus,u_dns_plus,'--k')
ax_dns_log_plus.semilogx(y_dns_p ,u_dns_p)

log_line = lambda x: np.log(x)/0.41 + 5.2
y_vec = np.array([y_dns_p[-80],y_dns_p[-10]])
ax_dns_log_plus.plot(y_vec,log_line(y_vec)), 
ax_dns_log_plus.plot(25,log_line(25),'xk'), 

plt.show()
