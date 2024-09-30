import numpy as np
import numba as nb
"""
This file transforms the Hamiltonian to the exciton basis to facilitate the construction of the Redfield tensor

"""

# ====== getting the unitary transformation matrix and the energy gaps for Redfield ======
@nb.jit(nopython=True, fastmath=True)
def get_u_Omega_uv(hams, NStates):   

    # get the eigen energies and eigen states of Hs and sort ascendingly
    # EigVals, EigVecs = np.linalg.eigh(hams)
    En, U = np.linalg.eigh(hams)
    # sortinds = np.argsort(EigVals)
    # U = EigVecs[:,sortinds]
    U = U.astype(np.complex128)
    # En = EigVals[sortinds]

    # record the energy differences, i.e., present omega_ab
    w_uv = np.zeros((NStates, NStates), dtype = np.complex128)
    for i in range(NStates):
        for j in range (NStates):
            w_uv[i, j] = En[i] - En[j]

    return U, En, w_uv
