import torch
import copy
from scoring.score_dataset import run_decoder
import utils
import os
import numpy as np
from scoring import score_model
from generating import generating_model as gen_model
import scoring
from global_params import params


def train_score_model_with_generator(scoring_model: score_model.ScoringTransformer, scoring_model_copy: score_model.ScoringTransformer,
                                     generating_model: gen_model.GeneratingModel, plot_loss=None, skip_testing=False):
    n = (generating_model.n_bits + generating_model.n_checks) * 2

    def ge(): return utils.sample_iid_error(n)
    err = ge()
    def gc(): return generating_model.generate_sample(scoring_model_copy, err)

    scoring.score_training.main_training_loop(
        scoring_model, ge, gc, params['scoring_model_save_path'], params['n_score_training_per_epoch_genetic'],  plot_loss, skip_testing=skip_testing)


def main(qubit_err_probs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(utils.get_numb_type())

    scoring_model = torch.load(os.path.join(
        params['scoring_model_save_path']))

    generating_model = None
    generating_model = gen_model.GeneratingModel(
        device=device
    )
    n = (params['n_data_qubits'] + params['n_check_qubits']) * 2
    pc, _, _, _ = generating_model.generate_sample(
        scoring_model, utils.sample_iid_error(n))
    n_sample = 20_000
    n_succ = run_decoder(pc, n_sample, qubit_err_probs)
    print(f"{n_succ}/{n_sample}: {n_succ/n_sample}")


if __name__ == '__main__':
    main()
