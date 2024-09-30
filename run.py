import sys
import time
import numba as nb
import sys, os

parallel = True
Cpus     = 50
if (parallel == True):
    sys.path.append('../Models')
    sys.path.append('../Methods')
    sys.path.append(os.popen("pwd").read().split("/tmpdir")[0]) # INCLUDE PARENT DIRECTORY WHICH HAS METHOD AND MODEL FILES
    JOBID = str(os.environ["SLURM_ARRAY_JOB_ID"])               # GET ID OF THIS JOB
    TASKID = str(os.environ["SLURM_ARRAY_TASK_ID"])             # GET ID OF THIS TASK WITHIN THE ARRAY 

    nrank = int(TASKID)                                         # JOD ID FOR A JOB 
    size  = Cpus                                            # TOTAL NUMBER OF PROCESSOR AVAILABLE
else:
    sys.path.append('./Models')
    sys.path.append('./Methods')
    nrank = 0
    size  = 1

@nb.jit(nopython=True, fastmath=True)
def initR(ωj, β):
    σP = np.sqrt(ωj / (2 * np.tanh(0.5*β*ωj)))
    σx = σP/ωj

    x = np.random.normal(loc=0.0, scale=1.0, size= len(ωj)) * σx
    return x

@nb.jit(nopython=True, fastmath=True)
def get_Hs(ε, Δ):

    Hs = np.zeros((2,2))

    Hs[0, 0] = ε
    Hs[1, 1] = - ε
    Hs[0, 1] = Hs[1, 0] = Δ

    return Hs

@nb.jit(nopython=True, fastmath=True)
def get_Qs(NStates):
    Qs = np.zeros((NStates, NStates), dtype = np.complex128)
    Qs[0, 0] = 1.0
    Qs[1, 1] = - 1.0
    return Qs
# ======== Runge-Kutta-4 integrator ========
@nb.jit(nopython=True, fastmath=True)
def rk4(f, y0, tt, h, qmds_ad, En, w_uv, par):

    k1 = h * f(tt, y0, qmds_ad, En, w_uv, par)
    k2 = h * f(tt + h / 2, y0 + k1 / 2, qmds_ad, En, w_uv, par)
    k3 = h * f(tt + h / 2, y0 + k2 / 2, qmds_ad, En, w_uv, par)
    k4 = h * f(tt + h, y0 + k3, qmds_ad, En, w_uv, par)
    
    return y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6

@nb.jit(nopython=True, fastmath=True)
def evolve(par):
    rho_save = np.zeros((nData, 4), dtype = np.complex128)
    x_slow = initR(par.ω_slow, beta)
    λ_cl   = np.sum(x_slow * par.c_slow)
    ε      = par.ε + λ_cl

    hams = get_Hs(ε, par.Δ)

    qmds = np.zeros((nmod, NStates, NStates), dtype = np.complex128)
    qmds[0, :, :] = get_Qs(NStates)
    # featured for Redfield
    U, En, w_uv = get_u_Omega_uv(hams, NStates)
    En = En.astype(np.complex128)
    qmds_ad = np.zeros((nmod, NStates, NStates), dtype = np.complex128)
    U = U.copy()
    qmds = qmds.copy()
    for i in range(nmod):           
        qmds_ad[i, :, :] = U.T.conjugate() @ qmds[i, :, :] @ U
    # ======== switching rho0 to the adiabatic representation ======== 
    rho0_ad = U.T.conjugate() @ rho0 @ U
    # ======== initial density matrix ======== 
    rhot_ad = rho0_ad
    iskip = 0
    # Redfield tensors initialization (for different implementations)
    par.R_ten   = np.zeros((NStates, NStates, NStates, NStates), dtype=np.complex128)
    for i in range(NSteps - 1):
        if (i % nskip == 0):
            U = U.copy()
            rhot_ad = rhot_ad.copy()
            rhot_d = U @ rhot_ad @ U.T.conjugate()
            rho_save[iskip, :] = rhot_d.reshape(4)
            iskip += 1
        rhot_ad = rk4(func, rhot_ad, i * dt, dt, qmds_ad, En, w_uv, par)
    return rho_save

start_time = time.time()


from Methods.Redfield_Mat_2 import func             # the fastest one, use this
from Models.Frozen_modes import parameters
from Models.exc_basis import get_u_Omega_uv 
from Models.constants import *

# ==============================================================================================
#                                       par passing     
# ==============================================================================================

# ======== Passing the model-specific system and bath par ========
par = parameters()
NStates = par.NStates    # number of states in the reduced system part
beta = par.β             # inverse temperature
coeff = par.coeff        # bath oscillator coupling coefficients
ω = par.ω                # bath oscillator frequencies
ndof = par.ndof          # number of bath DOFs
nmod = par.nmod          # number of dissipation modes
rho0 = par.rho0          # initial density matrix under the diabatic representation

# ======== Dynamics control par ========
t = par.t                # total propagation time
dt = par.dt              # integration time step & bare-bath TCF discretization spacing
NSteps = int(t / dt)            # number of steps
nskip = par.nskip        # step intervals to print data
dd = 0 if (NSteps % nskip == 0) else 1
N_out = NSteps//nskip + dd      # number of data points to record
traj = 10000
if NSteps%nskip == 0:
    nData = NSteps // nskip + 0
else :
    nData = NSteps // nskip + 1
Sim_time = np.array([(x * dt) for x in range(NSteps)])
times = Sim_time[::nskip]
r_save = np.zeros((nData, 4), dtype = np.complex128)

# ==============================================================================================
#                                       Initialization     
# ==============================================================================================
tot_Tasks = traj
NTasks = tot_Tasks//size
NRem = tot_Tasks - (NTasks*size)
TaskArray = [i for i in range(nrank * NTasks , (nrank+1) * NTasks)]
for i in range(NRem):
    if i == nrank: 
        TaskArray.append((NTasks*size)+i)
TaskArray = np.array(TaskArray)                                  # CONTAINS THE NUMBER OF TRAJECTORIES ASSIGNED TO EACH JOB

for k in range(len(TaskArray)):
    r_save += evolve(par)

end_time = time.time()
run_time = end_time - start_time

if (parallel == True):
    np.savetxt(f'../data/rho_traj_{nrank}.txt', np.c_[times, r_save/len(TaskArray)])
else:
    np.savetxt(f'./data/rho_traj_{nrank}.txt', np.c_[times, r_save/len(TaskArray)])

print("time consumption", run_time/60, "minutes")