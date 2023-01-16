import torch
import utils
import os
import numpy as np
from scoring import score_model
from generating import generating_model as gen_model
import scoring
from global_params import params


def initialize_scoring_model(device, plot_loss=None):
    def gc(): return scoring.initial_code_sampling.generate_code(params)
    _sample_code, _, _, _ = gc()
    sample_code = _sample_code.get_classical_code()
    n = sample_code.shape[-1]
    k = n - sample_code.shape[-2]
    def ge(): return utils.sample_iid_error(n)
    N_dec = 3  # CHanged from 6
    h = 4  # changed from 8...
    d_model = 40  # default is 32 but we are adding parity check info...
    model = score_model.ScoringTransformer(
        params['n_data_qubits'], params['n_check_qubits'], h, d_model, N_dec, device, dropout=0).to(device)
    scoring.score_training.main_training_loop(
        model, ge, gc, params['scoring_model_save_path'], plot_loss)
    return model
    # model = torch.load(os.path.join(save_path, 'best_model'))


def main(plot_loss=None, load_saved_scoring_model=False, load_saved_generating_model=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(utils.get_numb_type())
    scoring_model = None
    if not load_saved_scoring_model:
        scoring_model = initialize_scoring_model(device, plot_loss)
    else:
        scoring_model = torch.load(os.path.join(
            params['scoring_model_save_path']))
    generating_model = None
    if not load_saved_generating_model:
        gen_model.GeneratingModel(
            params['n_data_qubits'], params['n_check_qubits'], params['deg_bits'], params['deg_phase'], params['deg_cc']
        )
    return scoring_model


if __name__ == '__main__':
    main()
