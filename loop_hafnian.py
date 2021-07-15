import numpy as np 
import numba 
from _loop_hafnian_subroutines import (
    precompute_binoms,
    nb_ix,
    matched_reps,
    find_kept_edges,
    f_loop,
    f_loop_odd,
    get_submatrices,
    eigvals
    )

@numba.jit(nopython=True, parallel=True, cache=True)
def _calc_loop_hafnian(A, D, edge_reps,
        oddloop=None, oddV=None,
        glynn=True):

    n = A.shape[0]
    N = 2 * edge_reps.sum() # number of photons
    if oddloop is not None:
        N += 1 
    if glynn and (oddloop is None):
        steps = ((edge_reps[0] + 2) // 2) * np.prod(edge_reps[1:] + 1)
    else:
        steps = np.prod(edge_reps + 1)

    # precompute binomial coefficients 
    max_binom = edge_reps.max() + 1
    binoms = precompute_binoms(max_binom)

    H = np.complex128(0) #start running total for the hafnian

    for j in numba.prange(steps):

        kept_edges = find_kept_edges(j, edge_reps)
        edge_sum = kept_edges.sum()

        binom_prod = 1.
        for i in range(n//2):
            binom_prod *= binoms[edge_reps[i], kept_edges[i]]
        
        if glynn:
            kept_edges = 2 * kept_edges - edge_reps

        AX_S, XD_S, D_S, oddVX_S = get_submatrices(kept_edges, A, D, oddV)

        E = eigvals(AX_S) # O(n^3) step

        prefac = (-1.) ** (N//2 - edge_sum) * binom_prod

        if oddloop is not None:
            Hnew = prefac * f_loop_odd(E, AX_S, XD_S, D_S, N, oddloop, oddVX_S)[N]
        else:
            if glynn and kept_edges[0]==0:
                prefac *= 0.5
            Hnew = prefac * f_loop(E, AX_S, XD_S, D_S, N)[N//2]

        H += Hnew

    if glynn:
        if oddloop is None:
            H = H * 0.5 ** (N//2 - 1)
        else:
            H = H * 0.5 ** (N//2)

    return H

def loop_hafnian(A, D=None, reps=None,
    glynn=True):
    """
    calculate loop hafnian of NxN matrix, A
    reps is an optional N length vector which tells us
    how many times each row/column is repeated
    """
    # A is the matrix without repeats, reps contains the number of repeats for
    # each index

    n = A.shape[0]

    if reps is None:
        reps = [1] * n
    if D is None:
        D = A.diagonal()

    N = sum(reps)

    if N == 0:
        return 1.

    if N == 1:
        return D[0]

    assert n == len(reps)

    assert D.shape[0] == n

    x, edge_reps, oddmode = matched_reps(reps)

    # make new A matrix and D vector using the ordering from above... 
    
    if oddmode is not None:
        oddloop = D[oddmode].astype(np.complex128)
        oddV = A[oddmode, x].astype(np.complex128)
    else:
        oddloop = None
        oddV = None

    Ax = A[np.ix_(x, x)].astype(np.complex128)
    Dx = D[x].astype(np.complex128)

    H = _calc_loop_hafnian(Ax, Dx, edge_reps, oddloop, oddV, glynn)
    return H

### compile code on some small instances ###
A = np.ones((3,3))
assert np.allclose(loop_hafnian(A), 4)
A = np.ones((4,4))
assert np.allclose(loop_hafnian(A), 10)
############################################