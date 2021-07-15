import numpy as np 
from thewalrus.quantum import Amat, Qmat
from strawberryfields.decomposition import williamson
from thewalrus.symplectic import loss
from loop_hafnian import loop_hafnian
from scipy.special import factorial
from scipy.linalg import block_diag

rng = np.random.default_rng()

def R_to_alpha(R, hbar=2):
    M = len(R) // 2
    # mean displacement of each mode
    alpha = (R[:M] + 1j * R[M:]) / np.sqrt(2 * hbar)
    return alpha

class ClickMIS:

    scale_factor = 1.

    def __init__(self, cov, N, lhaf_func=loop_hafnian, rank=0):

        self.lhaf_func = lhaf_func

        self.rank = rank

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

        Q = Qmat(T)
        A = Amat(Q, cov_is_qmat=True)
        B = A[:M, :M].conj()

        self.detQ = np.linalg.det(Q).real

        self.B = B 
        self.abs2_B = abs(B)**2
        self.sqrtW = sqrtW

        sample = self.sample_proposal()

        self.prob_proposal_chain = None 
        self.prob_target_chain = None 
        self.pattern_chain = None

        self.samples = []
        self.chain_patterns = []
        self.proposal_probs = []
        self.target_probs = []
        self.chain_outcomes = []
        self.prob_calc_times = []

        self.chain_started = False

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
        return sample, R, alpha

    def _apply_loss_target(self, x, R):
        Rx = R.copy()
        covx = self.T.copy()

        eta = 1 - x
        for i in range(self.M):
            Rx, covx = loss(Rx, covx, eta[i], i)
        return Rx, covx


    def _apply_loss_prop(self, x, alpha):
        abs2_B = self.abs2_B
        abs2_alpha = abs(alpha) ** 2
        eta = 1 - x

        Cx = abs2_B.copy()
        for j in range(self.M):
            for k in range(self.M):
                Cx[j,k] = eta[j] * eta[k] * abs2_B[j,k]
        dx = abs2_alpha.copy()
        for j in range(self.M):
            dx[j] = abs2_alpha[j]
            for k in range(self.M):
                dx[j] += x[k] * abs2_B[j,k]
            dx[j] *= eta[j]
        
        return dx, Cx

    def sample_proposal(self):
        sampleN = -1
        while sampleN != self.N:
            photon_pattern, R, alpha = self._sample_photons()
            click_modes = np.where(photon_pattern > 0)[0]
            sampleN = len(click_modes)

        x = np.zeros(self.M)
        x[click_modes] = rng.power(photon_pattern[click_modes])
        
        # dx, Cx = self._apply_loss_prop(x, alpha)
        Rx, covx = self._apply_loss_target(x, R)

        Dx, Sx = williamson(covx)
        Tx = Sx @ Sx.T 

        DmIx = Dx - np.eye(2 * self.M)
        DmIx[abs(DmIx) < 1e-10] = 0. #remove slightly negative values
        # Wx = Sx @ DmIx @ Sx.T
        sqrtWx = Sx @ np.sqrt(DmIx)
        delta_Rx = self.sample_R(sqrtWx)

        Rx2 = Rx + delta_Rx

        click_pattern = np.zeros(self.M, dtype=np.int16)
        click_pattern[click_modes] = 1
        return click_pattern, Rx2, Tx, x, alpha

    def proposal_prob(self, pattern, dx, Cx):
        lhaf = self.lhaf_func(Cx, dx, pattern).real 
        prefac = np.exp(-0.5 * Cx.sum() - dx.sum()).real
        return prefac * lhaf

    def sample_R(self, sqrtW=None):
        if sqrtW is None:
            sqrtW = self.sqrtW
        z = rng.normal(size=2*self.M)
        R = sqrtW @ z 
        return R

    def target_prob(self, pattern, Rx, Tx):
        alpha = R_to_alpha(Rx)
        Q = Qmat(Tx)
        A = Amat(Q, cov_is_qmat=True)
        B = A[:self.M, :self.M].conj()

        gamma = alpha - B @ alpha.conj()
        prefac_n = abs(np.exp(-0.5 * (np.linalg.norm(alpha) ** 2 - 
                                alpha.conj() @ B @ alpha.conj())))**2
        prefac_d = np.sqrt(np.linalg.det(Q).real)
        lhaf = self.lhaf_func(B, gamma, pattern)
        return (prefac_n * lhaf * lhaf.conjugate()).real  / prefac_d

    def start_chain(self):
        if len(self.samples) == 0:
            sample = self.sample_proposal()
            self.samples.append(sample)
        else:
            sample = self.samples[0]

        click_pattern, Rx2, Tx, x, alpha = sample
        dx, Cx = self._apply_loss_prop(x, alpha)

        t0 = perf_counter()
        proposal_prob = self.proposal_prob(click_pattern, dx, Cx)
        target_prob = self.target_prob(click_pattern, Rx2, Tx)
        prob_calc_time = perf_counter() - t0

        self.chain_patterns.append(click_pattern)
        self.prob_proposal_chain = proposal_prob
        self.prob_target_chain = target_prob 
        self.pattern_chain = click_pattern

        self.proposal_probs.append(proposal_prob)
        self.target_probs.append(target_prob)
        self.prob_calc_times.append(prob_calc_time)


    def update_chain(self, sample=None):
        if not self.chain_started:
            self.start_chain()
            self.chain_started = True

        if sample is None:
            sample = self.sample_proposal()
            self.samples.append(sample)

        click_pattern, Rx2, Tx, x, alpha = sample

        dx, Cx = self._apply_loss_prop(x, alpha)
        
        t0 = perf_counter()
        proposal_prob = self.proposal_prob(click_pattern, dx, Cx)
        target_prob = self.target_prob(click_pattern, Rx2, Tx)
        prob_calc_time = perf_counter() - t0

        accept_prob = min(1, 
            abs((self.prob_proposal_chain * target_prob) / 
            (self.prob_target_chain * proposal_prob)))

        rand = rng.random()

        if rand <= accept_prob:
            self.chain_patterns.append(click_pattern)
            self.prob_proposal_chain = proposal_prob
            self.prob_target_chain = target_prob 
            self.pattern_chain = click_pattern
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

    def generate_proposal_samples(self, chain_steps=1000):
        if not self.chain_started:
            self.start_chain()
            self.chain_started = True

        for i in range(chain_steps):
            sample = self.sample_proposal()
            self.samples.append(sample)

    def run_chain_from_samples():
        for sample in self.samples:
            self.update_chain(sample)

    def mean_N_proposal(self, trials=1000):

        N_sum = 0
        for i in range(trials):
            pattern, R, alpha = self._sample_photons()
            N_sum += np.count_nonzero(pattern)

        return N_sum / trials

    def set_scale_factor(self, trials=1000):

        mean_N = self.mean_N_proposal(trials)
        self.scale_factor = self.N / mean_N

    def output_events(self, burn_in, thinning_rate):
        return self.chain_patterns[burn_in::thinning_rate]
        