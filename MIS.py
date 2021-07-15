import numpy as np 
import abc
from thewalrus.quantum import Amat, Qmat
from strawberryfields.decompositions import williamson
from loop_hafnian import loop_hafnian
from scipy.special import factorial
from time import perf_counter

rng = np.random.default_rng()

def R_to_alpha(R, hbar=2):
    M = len(R) // 2
    # mean displacement of each mode
    alpha = (R[:M] + 1j * R[M:]) / np.sqrt(2 * hbar)
    return alpha

class MISChainBase(abc.ABC):
    def __init__(self, cov, N, lhaf_func=loop_hafnian, start_sampling=True):

        self.lhaf_func = lhaf_func

        self.cov = cov 
        M = cov.shape[0] // 2
        self.M = M
        assert cov.shape == (2 * M, 2 * M)
        self.N = N

        D, S = williamson(cov)
        T = S @ S.T 
        self.T = T

        DmI = D - np.eye(2 * M)
        DmI[abs(DmI) < 1e-10] = 0. #remove slightly negative values
        W = S @ DmI @ S.T
        sqrtW = S @ np.sqrt(DmI)

        A = Amat(T)
        B = A[:M, :M].conj()

        Q = Qmat(T)
        self.detQ = np.linalg.det(Q).real

        self.B = B 
        self.abs2_B = abs(B)**2
        self.sqrtW = sqrtW

        if start_sampling:
            pattern, R = self.sample_proposal()

            self.prob_proposal_chain = self.proposal_prob(pattern, R)
            self.prob_target_chain = self.target_prob(pattern, R)
            self.pattern_chain = pattern

            self.patterns = [pattern]
            self.chain_patterns = [pattern]
            self.proposal_probs = [self.prob_proposal_chain]
            self.target_probs = [self.prob_target_chain]
            self.chain_outcomes = []
            self.Rs = [R]
            self.prob_calc_times = []

    @abc.abstractmethod
    def sample_proposal(self):
        pass 

    @abc.abstractmethod
    def proposal_prob(self, pattern, R):
        pass 

    def sample_R(self):
        z = rng.normal(size=2*self.M)
        R = self.sqrtW @ z 
        return R

    def target_prob_lhaf_args(self, pattern, R):
        alpha = R_to_alpha(R)
        B = self.B
        gamma = alpha - B @ alpha.conj()
        return B, gamma
        
    def target_prob(self, pattern, R):  
        alpha = R_to_alpha(R)
        B = self.B
        gamma = alpha - B @ alpha.conj()
        prefac_n = abs(np.exp(-0.5 * (np.linalg.norm(alpha) ** 2 - 
                                alpha.conj() @ B @ alpha.conj())))**2
        prefac_d = np.prod(factorial(pattern), dtype=np.float64) * np.sqrt(self.detQ)
        lhaf = self.lhaf_func(B, gamma, pattern)
        return (prefac_n * lhaf * lhaf.conjugate()).real  / prefac_d

    def update_chain(self, pattern=None, R=None):
        if pattern is None or R is None:
            pattern, R = self.sample_proposal()
            self.patterns.append(pattern)
            self.Rs.append(R)
        
        t0 = perf_counter()
        proposal_prob = self.proposal_prob(pattern, R)
        target_prob = self.target_prob(pattern, R)
        prob_calc_time = perf_counter() - t0

        accept_prob = min(1, 
            (self.prob_proposal_chain * target_prob) / 
            (self.prob_target_chain * proposal_prob))

        rand = rng.random()

        if rand <= accept_prob:
            self.chain_patterns.append(pattern)
            self.prob_proposal_chain = proposal_prob
            self.prob_target_chain = target_prob 
            self.pattern_chain = pattern
            self.chain_outcomes.append(1)
        else:
            self.chain_patterns.append(self.pattern_chain)
            self.chain_outcomes.append(0)

        self.proposal_probs.append(proposal_prob)
        self.target_probs.append(target_prob)
        self.prob_calc_times.append(prob_calc_time)

    def run_chain(self, chain_steps=1000):
        for i in range(chain_steps):
            self.update_chain()

    def acceptance_rate(self):
        return np.mean(self.chain_outcomes)

    def mean_N_proposal(self, trials=1000):

        N_sum = 0
        for i in range(trials):
            pattern, R = self._sample_photons()
            N_sum += pattern.sum()

        return N_sum / trials

    def generate_proposal_samples(self, trials):
        for i in range(trials):
            pattern, R = self.sample_proposal()
            self.patterns.append(pattern)
            self.Rs.append(R)

    def run_chain_from_samples(self):
        for pattern, R in zip(self.patterns, self.Rs):
            self.update_chain(pattern, R)

class MIS_IPS(MISChainBase):

    scale_factor = 1. 

    def _sample_photons(self):
        R = self.sample_R()
        alpha = R_to_alpha(R)

        G = np.sqrt(self.scale_factor) * abs(alpha) ** 2
        C = self.scale_factor * self.abs2_B

        sample = rng.poisson(G)
        for j in range(self.M):
            sample[j] += 2 * rng.poisson(0.5 * C[j,j])
            for k in range(j+1, self.M):
                n = rng.poisson(C[j,k])
                sample[j] += n 
                sample[k] += n 
        return sample, R

    def sample_proposal(self):
        sampleN = -1
        while sampleN != self.N:
            pattern, R = self._sample_photons()
            sampleN = pattern.sum()
        return pattern, R

    def proposal_prob(self, pattern, R):
        alpha = R_to_alpha(R)
        G = np.sqrt(self.scale_factor) * abs(alpha) ** 2 
        C = self.scale_factor * self.abs2_B 
        prob = self.lhaf_func(C, G, pattern)
        prefac = np.exp(-np.sum(G))
        prob *= prefac / np.prod(factorial(pattern))
        return prob

    def set_scale_factor(self, trials=1000):

        mean_N = self.mean_N_proposal(trials)
        self.scale_factor *= self.N / mean_N
