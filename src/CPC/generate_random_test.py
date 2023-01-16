from CPC.cpc_code import *
from CPC import generate_random
import unittest
from bposd import bposd_decoder


def sample_noisy_codespace(n, p_failures):
    return (np.random.rand(n) <= p_failures).astype(np.uint32)

class TestCPCSimplifications(unittest.TestCase):
    def test_generate_random(self):
        code, _ = generate_random.random_cpc(60, 30, 4, 4, 2)
        classical_code = code.get_classical_code()
        print(classical_code)
        # print("CLASSICAL CODE", classical_code, classical_code.sum(axis=-1))
        n = classical_code.shape[-1]
        rho = np.ones((n)) * 0.02
        bpd = bposd_decoder(classical_code, channel_probs=rho,
                        bp_method="product_sum", osd_method="osd_e",  max_iter=40, osd_order=8)
        # TODO: refactor
        n_runs = 100
        n_succ = 0
        for i in range(n_runs):
            noise = sample_noisy_codespace(n, rho)
            synd = (classical_code @ noise) % 2
            print(noise.sum(), synd.sum())
            if sum(synd) == 0:
                n_succ += 1
            else:
                eded = bpd.decode(synd)
                eq = np.array_equal(eded, noise)
                n_succ += eq
        print("Success Rate", n_succ / n_runs)

