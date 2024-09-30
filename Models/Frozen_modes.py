import numpy as np
from numba.experimental import jitclass
from numba import int32, float64, complex128
from exc_basis import get_u_Omega_uv
import numba as nb
@nb.jit(nopython=True, fastmath=True)
def initR(ωj, β):
    σP = np.sqrt(ωj / (2 * np.tanh(0.5*β*ωj)))
    σx = σP/ωj

    x = np.random.normal(loc=0.0, scale=1.0, size= len(ωj)) * σx
    return x

@nb.jit(nopython=True, fastmath=True)
def coth(x):
    return (1 + np.exp(-2 * x)) / (1 - np.exp(-2 * x))

@nb.jit(nopython=True, fastmath=True)
def J_Ohm(alpha, ωc, ω):
    return np.pi/2 * alpha * ω * np.exp(-ω/ωc)

@nb.jit(nopython=True, fastmath=True)
def J_Drude(λ, γ, ω):
    return 2 * λ * γ * ω / (ω**2 + γ**2)

@nb.jit(nopython=True, fastmath=True)
def SplitF(ω, ωc):
  S = [(1 - (x/ωc)**2)**2 if x < ωc else 0 for x in ω]
  return S

@nb.jit(nopython=True, fastmath=True)
def bathParam(ω, J, ndof):     # for bath descritization
    cj = np.zeros(( ndof ))
    ωj = np.zeros(( ndof ))

    dω = ω[1] - ω[0]
    
    Fω = np.zeros(len(ω))
    for i in range(len(ω)):
        Fω[i] = (4/np.pi) * np.sum(J[:i]/ω[:i]) * dω

    λs =  Fω[-1]
    for i in range(ndof):
        costfunc = np.abs(Fω-(((float(i)+0.5)/float(ndof))*λs))
        m = np.argmin((costfunc))
        ωj[i] = ω[m]
    cj[:] = ωj[:] * ((λs/(2*float(ndof)))**0.5)

    return cj, ωj

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

@nb.jit(nopython=True, fastmath=True)
def get_rho0(NStates):

    rho0 = np.zeros((NStates, NStates), dtype = np.complex128)
    rho0[0, 0] = 1.0 + 0.0j

    return rho0


NStates = 2      # number of states
ε, Δ, β, γ, λ, ndof, t = 0, 1, 5.0, 0.25, 0.25, 300, 15     # Fig. a
# ε, Δ, β, γ, λ, ndof, t = 1, 1, 0.5, 0.25, 0.25, 300, 15     # Fig. b
# ε, Δ, β, γ, λ, ndof, t = 0, 1, 0.5, 0.25, 5.0, 300, 15    # Fig. c
# ε, Δ, β, γ, λ, ndof, t = 0, 1, 1.0, 1.0, 2.5, 300, 15    # Fig. d
ω_cut = np.max([γ, (Δ**2 + ε**2)**0.5/2])
ω_scan  = np.linspace(1E-10,100 * γ,50000)
S = SplitF(ω_scan, ω_cut)
J_slow = S * J_Drude(λ, γ, ω_scan)
J_fast = (np.ones(len(ω_scan)) - S) * J_Drude(λ, γ, ω_scan)
c_slow, ω_slow = bathParam(ω_scan, J_slow, ndof)
# propagation
dt = 0.0025             # integration time step
NSteps = int(t / dt)    # number of steps
nskip = 10              # interval for data recording

# produce the Hamiltonian, initial RDM

rho0 = get_rho0(NStates)

nmod = 1         # number of dissipation modes
C_ab = np.zeros((nmod, NSteps), dtype = np.complex128)    # bare-bath TCFs
coeff = np.zeros((nmod, ndof), dtype = np.complex128)
for n in range(nmod):
    coeff[n, :], ω  = bathParam(ω_scan, J_fast, ndof)
    for i in range(NSteps):
        C_ab[n, i] = 0.25 * np.sum((coeff[n, :]**2 / (2 * ω)) * (coth(β * ω / 2) * np.cos(ω * i * dt) - 1.0j * np.sin(ω * i * dt)))
    

spec = [
    ('NStates',             int32),
    ('ε',                 float64), 
    ('Δ',                 float64),
    ('β',                 float64), 
    ('γ',                 float64),
    ('λ',                 float64),
    ('ndof',                int32),
    ('t',                   int32),
    ('ω_slow',         float64[:]),
    ('c_slow',         float64[:]),
    ('dt',                float64),
    ('nskip',               int32),
    ('C_ab',      complex128[:,:]),
    ('R_ten', complex128[:,:,:,:]),
    ('ω',              float64[:]),
    ('coeff',        float64[:,:]),
    ('nmod',                int32),
    ('rho0',      complex128[:,:]), 
]

@jitclass(spec)
class parameters():
    def __init__(self):
        self.NStates = NStates      # number of states
        self.ε       = ε
        self.Δ       = Δ
        self.β       = β 
        self.γ       = γ
        self.λ       = λ
        self.ndof    = ndof
        self.t       = t 
        self.ω_slow  = ω_slow
        self.c_slow  = c_slow
        self.dt      = dt
        self.nskip   = nskip
        self.C_ab    = C_ab
        self.R_ten   = np.zeros((NStates, NStates, NStates, NStates), dtype=np.complex128)
        self.ω       = ω
        self.nmod    = nmod
        self.rho0    = rho0
# ====================================================================================================
    