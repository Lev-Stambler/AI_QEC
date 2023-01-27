from typing import Callable
import math
import json
import os
from multiprocessing import Pool, Process
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


def decode_random(params):
    n, rho, pc, block_size = params
    s = 0

    bpd = bposd_decoder(pc, channel_probs=rho,
                        bp_method="product_sum", osd_method="osd_e", osd_order=3, max_iter=5)
    for _ in range(block_size):
        noise = sample_noisy_codespace(n, rho)
        synd = (pc @ noise) % 2
        if sum(synd) == 0:
            s += 1
        else:
            eded = bpd.decode(synd)
            eq = np.array_equal(eded, noise)
            s += int(eq)
    return s

# TODO: can we parallelize this dramatically? I think yes
# TODO: move to utils
# Hmmm.... this is not working. Alternatively we just have the dataloader create the data upfront...


def run_decoder(pc, n_runs, p_fails, multiproc=False):
    n = pc.shape[1]
    rho = p_fails
    # if multiproc:
    #     block_size = 5_000
    #     assert n_runs // block_size == n_runs / block_size, "Number of runs must be multiple of block size"

    #     n_succ = 0
    #     n_pools = n_runs // block_size
    #     with Pool() as pool:
    #         result = pool.map(decode_random,
    #                           zip([n] * n_pools, [rho] * n_pools, [pc] * n_pools, [block_size] * n_pools))
    #         for s in result:
    #             n_succ += s
    #     return n_succ
    # else:
    return decode_random((n, rho, pc, n_runs))


def get_data_sample_file(dir, numb):
    return os.path.join(dir, f"d{numb}.json")


class ScoringDataset(torch.utils.data.Dataset):
    """Some Information about MyDataset"""

    def __init__(self, error_prob_sample: Callable[[], npt.NDArray], random_code_sample: Callable[[], npt.NDArray], raw_dataset_size, load_save_dir,
                 item_sample_size=None):
        self.error_prob = error_prob_sample
        self.random_code = random_code_sample
        self.raw_dataset_size = raw_dataset_size
        self.load_save_dir = load_save_dir
        self.samples = []
        self.unnormalized_sample_prob = []

        if item_sample_size is not None:
            self.item_sample_size = item_sample_size
        else:
            self.item_sample_size = params['n_decoder_rounds']

        print("Starting data generation")
        # Load the data
        if not os.path.exists(load_save_dir):
            os.mkdir(load_save_dir)

        scaling = 1.2
        for i in range(raw_dataset_size):
            if not os.path.exists(get_data_sample_file(load_save_dir, i)):
                self.generate_error_file(
                    get_data_sample_file(load_save_dir, i))

            d = self.load_file(i)
            if d["err_distr"][0] >= params["constant_error_rate_lower"] and \
                    d["err_distr"][0] <= params["constant_error_rate_upper"]:
                self.samples.append(d)
                self.unnormalized_sample_prob.append(math.e ** (scaling * d["err_rate"]))
                # self.unnormalized_sample_prob.append(0.01)

            if i % 1_000 == 0 and i != 0:
                print("Done generating sample", i)

        normalizing = sum(self.unnormalized_sample_prob)
        self.dataset_size = len(self.samples)
        self.normalized_probs = [
            p / normalizing for p in self.unnormalized_sample_prob]

        print("Done with data generation")
        super(ScoringDataset, self).__init__()

    def load_file(self, index):
        d = None
        with open(get_data_sample_file(self.load_save_dir, index), 'r') as openfile:
            # Reading from json file
            d = json.load(openfile)
        return d

    def __getitem__(self, _index):
        """
        Hmmm... clearly we have a problem. Maybe instead we randomly sample all where higher
        weighted ones get a higher prob of being sampled? Lets do something simple
        like normalize over e^((x/2)^2).
        """
        index = np.random.choice(
            np.arange(self.dataset_size), p=self.normalized_probs)
        print(f"Loading Index {index}")

        d = self.samples[index]
        e_type = np.float32 if torch.cuda.is_available() else np.double
        return (np.asarray(d['bit_adj']), np.asarray(d['phase_adj']), np.asarray(d['check_adj']), np.asarray(d['err_distr']).astype(e_type),
                d['err_rate'])

    def calculate_error_rate(self, code, error_prob):
        return run_decoder(code, self.item_sample_size, error_prob) / self.item_sample_size

    def generate_error_file(self, file_name):
        e = self.error_prob()
        cpc_code_pc, bit_adj, phase_adj, check_adj = self.random_code()
        err_rate = self.calculate_error_rate(cpc_code_pc, e)
        dictionary = {}
        dictionary['bit_adj'] = bit_adj
        dictionary['phase_adj'] = phase_adj
        dictionary['check_adj'] = check_adj
        dictionary['err_rate'] = err_rate
        dictionary['err_distr'] = e

        json_object = json.dumps(dictionary, cls=utils.NumpyArrayEncoder)

        # Writing to sample.json
        with open(file_name, "w") as outfile:
            outfile.write(json_object)

    def __len__(self):
        return self.dataset_size


np.max
