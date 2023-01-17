from typing import Callable
import torch
import numpy.typing as npt
from bposd import bposd_decoder
import numpy as np
import utils
from global_params import params


def gen_random_ldpc(n, k, deg_row):
    iden_left = np.eye(n - k)
    x = np.zeros((n-k, k))
    x[:, :deg_row - 1] = 1
    perm = np.stack([np.random.permutation(k) for _ in range((n - k))])
    x = np.take(x, perm)
    # TODO: above is wrong use first approach
    rand_ldpc_H = np.concatenate((iden_left, x), axis=1)
    G = utils.HtoG(rand_ldpc_H)
    return rand_ldpc_H.astype(np.int16), G.astype(np.int16)


def sample_noisy_codespace(n, p_failures):
    return (np.random.rand(n) <= p_failures).astype(np.uint16)


# TODO: can we parallelize this dramatically? I think yes
def run_decoder(pc, n_runs, p_fails):
    n = pc.shape[1]
    rho = p_fails
    bpd = bposd_decoder(pc, channel_probs=rho,
                        bp_method="product_sum", osd_method="osd_e", osd_order=3, max_iter=5)
    n_succ = 0
    for i in range(n_runs):
        noise = sample_noisy_codespace(n, rho)
        synd = (pc @ noise) % 2
        if sum(synd) == 0:
            n_succ += 1
        else:
            eded = bpd.decode(synd)
            eq = np.array_equal(eded, noise)
            n_succ += eq

    return n_succ

class ScoringDataset(torch.utils.data.Dataset):
    """Some Information about MyDataset"""

    def __init__(self, error_prob_sample: Callable[[], npt.NDArray], random_code_sample: Callable[[], npt.NDArray], dataset_size, item_sample_size=None):
        self.error_prob = error_prob_sample
        self.random_code = random_code_sample
        self.dataset_size = dataset_size
        if item_sample_size is not None:
            self.item_sample_size = item_sample_size
        else:
            self.item_sample_size = params['n_decoder_rounds']
        super(ScoringDataset, self).__init__()

    def __getitem__(self, index):
        cpc_code_pc, bit_adj, phase_adj, check_adj = self.random_code()
        e = self.error_prob()
        error_rate = self.calculate_error_rate(cpc_code_pc, e)
        # sample = {'code': H, 'error_probs': e, 'frame_error_rate': error_rate}

        e_type = np.float32 if torch.cuda.is_available() else np.double
        return (bit_adj, phase_adj, check_adj, e.astype(e_type), error_rate)
    
    def calculate_error_rate(self, code, error_prob):
        return run_decoder(code, self.item_sample_size, error_prob) / self.item_sample_size

    def __len__(self):
        return self.dataset_size
np.max