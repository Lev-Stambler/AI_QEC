import torch
from bposd import bposd_decoder
import numpy as np
from .. import utils


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
    pass


# TODO: can we parallelize this dramatically? I think yes
def run_decoder(pc_mat_systematic, n_runs, p_fails):
    n = pc_mat_systematic.shape[1]
    rho = p_fails
    bpd = bposd_decoder(pc_mat_systematic, channel_probs=rho,
                        bp_method="product_sum", osd_method="osd_e", osd_order=3, max_iter=5)
    n_succ = 0
    for i in range(n_runs):
        noise = sample_noisy_codespace(n, rho)
        synd = (pc_mat_systematic @ noise) % 2
        if sum(synd) == 0:
            n_succ += 1
        else:
            eded = bpd.decode(synd)
            eq = np.array_equal(eded, noise)
            n_succ += eq

    return n_succ

class ScoringInitDataset(torch.utils.data.Dataset):
    """Some Information about MyDataset"""

    def __init__(self, error_prob_sample: function, random_code_sample: function, dataset_size, item_sample_size=10_000):
        self.error_prob = error_prob_sample
        self.random_code = random_code_sample
        self.dataset_size = dataset_size
        super(ScoringInitDataset, self).__init__()

    def __getitem__(self, index):
        H, _ = self.random_code()
        e = self.error_prob()
        error_rate = self.calculate_error_rate(H, e)
        sample = {'code': H, 'error_probs': e, 'frame_error_rate': error_rate}

        return sample
    
    def calculate_error_rate(self, code, error_prob):
        return run_decoder(code, self.item_sample_size, error_prob) / self.item_sample_size

    def __len__(self):
        return self.dataset_size

if __name__ == "__main__":
    n = 40
    k = 10
    deg_row = 5
    p_error = 0.05
    def ldpc_sample():
        return gen_random_ldpc(n, k, deg_row)
    def error_sample():
       return (np.random.rand(n) <= p_error).astype(np.uint16)
    d = ScoringInitDataset(error_sample, ldpc_sample, 1_000)
    for i in range(10):
        print(f"Sample {i + 1}: {d.__getitem__(i)}")