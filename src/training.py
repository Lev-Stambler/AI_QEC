import torch
import utils
import os
import numpy as np
from scoring import score_model
from generating import generating_model as gen_model
import scoring
from global_params import params


def initialize_scoring_model(device, plot_loss=None, skip_testing=False):
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
        model, ge, gc, params['scoring_model_save_path'], plot_loss, skip_testing=skip_testing)
    return model
    # model = torch.load(os.path.join(save_path, 'best_model'))


def train_score_model_with_generator(scoring_model: score_model.ScoringTransformer,
                                     generating_model: gen_model.GeneratingModel, plot_loss=None, skip_testing=False):
    n = (generating_model.n_bits + generating_model.n_checks) * 2

    def ge(): return utils.sample_iid_error(n)
    err = ge()
    def gc(): return generating_model.generate_sample(scoring_model, err)
    
    scoring.score_training.main_training_loop(
        scoring_model, ge, gc, params['scoring_model_save_path'], plot_loss, skip_testing=skip_testing)


def main(plot_loss=None, load_saved_scoring_model=False, load_saved_generating_model=False, skip_testing=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(utils.get_numb_type())
    scoring_model = None
    if not load_saved_scoring_model:
        scoring_model = initialize_scoring_model(device, plot_loss, skip_testing=skip_testing)
    else:
        scoring_model = torch.load(os.path.join(
            params['scoring_model_save_path']))
    generating_model = None
    if not load_saved_generating_model:
        generating_model = gen_model.GeneratingModel(
            params['n_data_qubits'], params['n_check_qubits'],  params['deg_phase'], params['deg_bit'], params['deg_check_to_check'], device=device
        )
    else:
        # TODO: load generating model??
        pass
    for genetic_epoch in range(params['n_genetic_epochs']):
        print(f"Starting epoch #{genetic_epoch + 1}")
        train_score_model_with_generator(
            scoring_model, generating_model, plot_loss, skip_testing=skip_testing)

    return scoring_model


if __name__ == '__main__':
    main()
