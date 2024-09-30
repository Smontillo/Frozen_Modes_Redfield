import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# ==============================================================================================
#                                         data reading     
# ==============================================================================================
comp = np.loadtxt('./Benchmark/Fig_2a.txt')
data = np.loadtxt("./data/rho_traj_0.txt", dtype = 'complex')
ρ = np.zeros((len(data[:,0]), len(data[0,:])))
proc = 50

for k in range(proc):
    ρ += np.real(np.loadtxt(f"./data/rho_traj_{k}.txt", dtype = 'complex'))

ρ /= proc

fig, ax = plt.subplots(figsize = (4.5,4.5))
ax.plot(ρ[:, 0], ρ[:, 1] - ρ[:, 4], "-", linewidth = 2.0, color = 'red', label = "sigmaz")
ax.plot(comp[:, 0], comp[:, 1] , ls = " ", marker = 'o', linewidth = 1.0, color = 'blue', label = 'Andres')

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='major', length=12, labelsize = 13, direction = 'in')
ax.tick_params(which='minor', length=5, direction = 'in')
ax.legend(loc=0, frameon = False, fontsize = 9, handlelength=1, labelspacing = 0.2)
ax.set_xlim(0,15)
ax.set_ylabel(r'$\langle \sigma_z \rangle$', fontsize = 20)
ax.set_xlabel(r'Time ($\Delta^{-1}$)', fontsize = 20)
plt.savefig('./Images/sigmaz.png', dpi = 300,  bbox_inches='tight')
plt.close()

# # plot coherence
# plt.plot(data2[:, 0], data2[:, 3], "-", linewidth = 2.0, color = 'red', label = "coherence_real")
# plt.plot(comp[:, 0], data2[:, 3], "-.", linewidth = 1.0, color = 'blue')

# plt.legend(frameon = False)
# # plt.xlim(0, 12)
# # plt.ylim(-2, 2)
# # plt.show()
# plt.savefig('./Images/coherence.png', dpi = 300)
# plt.close()
