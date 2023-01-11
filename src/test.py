import numpy as np
from scoring.data_generation import ScoringInitDataset, gen_random_ldpc


n = 40
k = 10
deg_row = 5
p_error = 0.05
def ldpc_sample():
    return gen_random_ldpc(n, k, deg_row)
def error_sample():
   return np.ones(n) * p_error
d = ScoringInitDataset(error_sample, ldpc_sample, 1_000)
for i in range(10):
    print(f"Sample {i + 1}: {d.__getitem__(i)}")