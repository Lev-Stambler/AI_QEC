import numpy as np
import random
from global_params import params


# TODO: here |x| = |z|...
def random_cpc(n_bits=params['n_data_qubits'], n_checks=params['n_check_qubits']):
    bit_adj = np.zeros((n_bits, n_checks), dtype=np.int16)
    phase_adj = np.zeros((n_bits, n_checks), dtype=np.int16)
    check_check_adj = np.zeros((n_checks, n_checks), dtype=np.int16)

    for i in range(n_bits):
        # For each PC, generate the degrees
        deg_phase = random.randint(
            params['deg_phase_lower'], params['deg_phase_upper'])
        deg_bit = random.randint(
            params['deg_bit_lower'], params['deg_bit_upper'])
        deg_cc = random.randint(
            params['deg_check_to_check_lower'], params['deg_check_to_check_upper'])

        check_idx = np.random.permutation(n_checks)[:deg_phase]
        for c in check_idx:
            phase_adj[i, c] = 1

        check_idx = np.random.permutation(n_checks)[:deg_bit]
        for c in check_idx:
            bit_adj[i, c] = 1

    for c1 in range(n_checks):
        check_idx = None
        while True:
            check_idx = np.random.permutation(n_checks)[:deg_cc]
            if c1 not in check_idx:
                break

        for c2 in check_idx:
            check_check_adj[c1, c2] = 1
            check_check_adj[c2, c1] = 1

    phase_check = bit_adj.T.dot(phase_adj) ^ check_check_adj
    pc = np.concatenate([phase_adj.T, phase_check,
                         bit_adj.T, np.eye(n_checks)], axis=-1)

    return pc, bit_adj, phase_adj, check_check_adj
