import numpy as np 
from loop_hafnian_batch import loop_hafnian_batch
from loop_hafnian_batch_gamma import loop_hafnian_batch_gamma
from scipy.special import factorial
from strawberryfields.decompositions import williamson
from thewalrus.quantum import (
    Amat,
    Qmat,
    photon_number_mean_vector, 
    mean_clicks,
    reduced_gaussian
    )

def decompose_cov(cov):
    m = cov.shape[0] // 2
    D, S = williamson(cov)
    T = S @ S.T 
    DmI = D - np.eye(2*m)
    DmI[abs(DmI) < 1e-11] = 0. # remove slightly negative values
    sqrtW = S @ np.sqrt(DmI)
    return T, sqrtW

def mu_to_alpha(mu, hbar=2):
    M = len(mu) // 2
    # mean displacement of each mode
    alpha = (mu[:M] + 1j * mu[M:]) / np.sqrt(2 * hbar)
    return alpha

def invert_permutation(p):
    s = np.empty_like(p, dtype=int)
    s[p] = np.arange(p.size, dtype=int)
    return s

def photon_means_order(mu, cov):
    means = photon_number_mean_vector(mu, cov)
    order = [x for _, x in sorted(zip(means, range(len(means))))]
    return np.asarray(order)

def click_means_order(cov):

    M = cov.shape[0] // 2 
    mu = np.zeros(2*M)

    means = np.zeros(M)

    for i in range(M):
        mu_i, cov_i = reduced_gaussian(mu, cov, [i])
        means[i] = mean_clicks(cov_i)

    order = [x for _, x in sorted(zip(means, range(len(means))))]
    return np.asarray(order)

def get_samples(mu, cov, cutoff=10, n_samples=10):
    M = cov.shape[0] // 2

    order = photon_means_order(mu, cov)
    order_inv = invert_permutation(order)
    oo = np.concatenate((order, order+M))

    mu = mu[oo]
    cov = cov[np.ix_(oo, oo)]

    T, sqrtW = decompose_cov(cov)
    chol_T_I = np.linalg.cholesky(T+np.eye(2*M))   
    B = Amat(T)[:M,:M] 
    det_outcomes = np.arange(cutoff+1)

    for i in range(n_samples):
        det_pattern = np.zeros(M, dtype=int)
        pure_mu = mu + sqrtW @ np.random.normal(size=2*M)
        pure_alpha = mu_to_alpha(pure_mu)
        heterodyne_mu = pure_mu + chol_T_I @ np.random.normal(size=2*M)
        heterodyne_alpha = mu_to_alpha(heterodyne_mu)
       
        gamma = pure_alpha.conj() + B @ (heterodyne_alpha - pure_alpha)
        for mode in range(M):
            m = mode + 1
            gamma -= heterodyne_alpha[mode] * B[:, mode]
            lhafs = loop_hafnian_batch(B[:m,:m], gamma[:m], det_pattern[:mode], cutoff)
            probs = (lhafs * lhafs.conj()).real / factorial(det_outcomes)
            norm_probs = probs.sum()
            probs /= norm_probs 

            det_outcome_i = np.random.choice(det_outcomes, p=probs)
            det_pattern[mode] = det_outcome_i

        yield det_pattern[order_inv]

def get_heterodyne_fanout(alpha, fanout):
    M = len(alpha)

    alpha_fanout = np.zeros((M, fanout), dtype=np.complex128)
    for j in range(M):
        alpha_j = np.zeros(fanout, dtype=np.complex128)
        alpha_j[0] = alpha[j] # put the coherent state in 0th mode 
        alpha_j[1:] = (np.random.normal(size=fanout-1) +
                 1j * np.random.normal(size=fanout-1))

        alpha_fanout[j,:] = np.fft.fft(alpha_j, norm='ortho')

    return alpha_fanout

def get_samples_click(mu, cov, cutoff=1, fanout=10, n_samples=10):

    M = cov.shape[0] // 2

    order = photon_means_order(mu, cov)
    order_inv = invert_permutation(order)
    oo = np.concatenate((order, order+M))

    mu = mu[oo]
    cov = cov[np.ix_(oo, oo)]
    T, sqrtW = decompose_cov(cov)
    chol_T_I = np.linalg.cholesky(T+np.eye(2*M))   
    B = Amat(T)[:M,:M] / fanout

    det_outcomes = np.arange(cutoff+1)

    for i in range(n_samples):
        det_pattern = np.zeros(M, dtype=int)
        click_pattern = np.zeros(M, dtype=np.int8)
        fanout_clicks = np.zeros(M, dtype=int)

        pure_mu = mu + sqrtW @ np.random.normal(size=2*M)
        pure_alpha = mu_to_alpha(pure_mu)
        het_mu = pure_mu + chol_T_I @ np.random.normal(size=2*M)
        het_alpha = mu_to_alpha(het_mu)

        het_alpha_fanout = get_heterodyne_fanout(het_alpha, fanout)
        het_alpha_sum = het_alpha_fanout.sum(axis=1)

        gamma = (pure_alpha.conj() / np.sqrt(fanout) + 
                    B @ (het_alpha_sum - np.sqrt(fanout) * pure_alpha))
        gamma_fanout = np.zeros((fanout, M), dtype=np.complex128)

        for mode in range(M):
            gamma_fanout[0,:] = gamma - het_alpha_fanout[mode, 0] * B[:, mode]
            for k in range(1, fanout):
                gamma_fanout[k,:] = gamma_fanout[k-1,:] - het_alpha_fanout[mode,k] * B[:,mode]
            lhafs = loop_hafnian_batch_gamma(B[:mode+1,:mode+1], gamma_fanout[:,:mode+1], 
                                            det_pattern[:mode], cutoff)
            probs = (lhafs * lhafs.conj()).real / factorial(det_outcomes)

            for k in range(fanout):
                gamma = gamma_fanout[k,:]
                probs_k = probs[k,:] / probs[k,:].sum()
                det_outcome = np.random.choice(det_outcomes, p=probs_k)
                det_pattern[mode] += det_outcome
                if det_outcome > 0:
                    click_pattern[mode] = 1
                    fanout_clicks[mode] = k
                    break 

        yield click_pattern[order_inv]
