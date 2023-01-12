import torch
import utils
import os
import numpy as np
from scoring import score_model
import scoring
from global_params import params


def initialize(device):
    gc = lambda: scoring.initial_code_sampling.generate_code(params)
    sample_code = gc()
    n = sample_code.shape[-1]
    k = n - sample_code.shape[-2]
    ge = lambda: utils.sample_iid_error(n)
    N_dec = 3  # CHanged from 6
    h = 4  # changed from 8...
    d_model = 40  # default is 32 but we are adding parity check info...
    model = score_model.ScoringTransformer(
        n, k, h, d_model, N_dec, dropout=0).to(device)
    scoring.score_training.main_training_loop(model, ge, gc, 'best_model')
    return model
    # model = torch.load(os.path.join(save_path, 'best_model'))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize(device)