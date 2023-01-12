import torch
import os
import numpy as np
from scoring import score_model
import scoring


def initialize(score_model: score_model.ScoringTransformer):
    scoring.score_training
    N_dec = 3  # CHanged from 6
    h = 4  # changed from 8...
    d_model = 40  # default is 32 but we are adding parity check info...
    model = score_model.ScoringTransformer(
        n, k, h, d_model, N_dec, dropout=0).to(device)
    scoring.score_training
    # model = torch.load(os.path.join(save_path, 'best_model'))
