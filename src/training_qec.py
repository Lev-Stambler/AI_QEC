import torch
import json
import copy
from scoring.score_dataset import run_decoder
import utils
import os
import numpy as np
from scoring import score_model
from generating import generating_model as gen_model
import scoring
from global_params import params


def initialize_scoring_model(device, plot_loss=None, scoring_model=None, initialize_epoch_start=1):
    def gc(): return scoring.initial_code_sampling.generate_code()
    sample_code, _, _, _ = gc()
    n = sample_code.shape[-1]
    k = n - sample_code.shape[-2]
    def ge(): return utils.sample_iid_error(n)
    N_dec = 6  # CHanged from 6
    h = 8  # changed from 8
    d_model = 32
    model = score_model.ScoringTransformer(
        params['n_data_qubits'], params['n_check_qubits'], h, d_model, N_dec, device, dropout=0).to(device) if scoring_model is None else scoring_model
    scoring.score_training.main_training_loop(
        "initialization", model, ge, gc, utils.get_best_scoring_model_path(
        ), params['n_score_training_per_epoch_initial'],
        epochs=params['n_score_epochs_initial'],
        plot_loss=plot_loss,
        epoch_start=initialize_epoch_start)
    return model
    # model = torch.load(os.path.join(save_path, 'best_model'))


def evaluate_performance(scoring_model: score_model.ScoringTransformer, gen_model: gen_model.GeneratingModel, low_p: list[int], epoch, is_init_epoch, n_tests=100_000, averaging_samples=100, eval_file=None):
    n = (params['n_data_qubits'] + params['n_check_qubits']) * 2
    eval_file = eval_file if eval_file is not None else utils.get_eval_path()

    json_object = None
    print(
        f"Starting to evaluate performance for epoch {epoch} with file {eval_file}")
    if os.path.exists(eval_file):
        with open(eval_file, 'r') as openfile:
            # Reading from json file
            json_object = json.load(openfile)
    else:
        json_object = {}

    overall_name = f"{'init' if is_init_epoch else 'genetic'}_epoch_{epoch}"
    json_object[overall_name] = {}
    best_low_p_succ_rate = [0.0] * len(low_p)
    best_low_p_pcs = [[]] * len(low_p)
    for _ in range(averaging_samples):
        cum_succ_rate = 0
        for i in range(len(low_p)):
            p = low_p[i]
            err = np.ones(n) * p
            pc, _, _, _ = gen_model.generate_sample(
                scoring_model, err, mutate=False)
            r = run_decoder(pc, err, multiproc=False)
            if r > best_low_p_succ_rate[i]:
                best_low_p_succ_rate[i] = r
                best_low_p_pcs = pc

        json_object[overall_name][f"p_{p}"] = cum_succ_rate / \
            averaging_samples

        for p, best_wsr, best_pc in zip(low_p, best_low_p_succ_rate, best_low_p_pcs):
            json_object[overall_name][f"low_p_best_{p}"] = best_wsr
            json_object[overall_name][f"low_p_best_{p}_pc"] = best_pc

    with open(eval_file, "w") as outfile:
        print("WRITING", json_object)
        json.dump(json_object, outfile, cls=utils.NpEncoder)
    print(f"Done evaluating performance for epoch {epoch}")


def train_score_model_with_generator(genetic_epoch, scoring_model: score_model.ScoringTransformer, scoring_model_copy: score_model.ScoringTransformer,
                                     generating_model: gen_model.GeneratingModel, plot_loss=None):
    n = (params['n_data_qubits'] + params['n_check_qubits']) * 2

    def ge(): return utils.sample_iid_error(n)
    err = ge()
    def gc(): return generating_model.generate_sample(scoring_model_copy, err)

    scoring.score_training.main_training_loop(f"gen_epoch_{genetic_epoch}",
                                              scoring_model, ge, gc, utils.get_best_scoring_model_path(
                                              ), params['n_score_training_per_epoch_genetic'],
                                              epochs=params['n_score_epochs_genetic'], plot_loss=plot_loss
                                              )


def main(plot_loss=None, load_saved_scoring_model=False, load_saved_generating_model=False, skip_initialization_training=False, skip_eval=False, initialize_epoch_start=1, genetic_epoch_start=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(utils.get_numb_type())
    scoring_model = None
    if load_saved_scoring_model:
        scoring_model = torch.load(os.path.join(
            utils.get_best_scoring_model_path()))

    generating_model = None
    if not load_saved_generating_model:
        generating_model = gen_model.GeneratingModel(
            device=device
        )
    else:
        raise "No option to load a saved generating model"

    p_eval_range = params['eval_p_range']

    if not skip_initialization_training:
        # # Get a baseline after 1 epoch
        # if initialize_epoch_start > 1:
        #     evaluate_performance(scoring_model, generating_model, p_eval_range, params['eval_p_range'] , -1)
        scoring_model = initialize_scoring_model(
            device, plot_loss, scoring_model=scoring_model, initialize_epoch_start=initialize_epoch_start)
    for genetic_epoch in range(genetic_epoch_start, params['n_genetic_epochs']):
        if not skip_eval:
            evaluate_performance(scoring_model, generating_model, p_eval_range,
                                 genetic_epoch, True and initialize_epoch_start == 1)
        print(f"Starting epoch #{genetic_epoch + 1}")
        scoring_copied = copy.deepcopy(scoring_model)
        train_score_model_with_generator(genetic_epoch,
                                         scoring_model, scoring_copied, generating_model, plot_loss)
        del scoring_copied
    if not skip_eval:
        evaluate_performance(scoring_model, generating_model,
                             p_eval_range, genetic_epoch, False)

    return scoring_model


if __name__ == '__main__':
    main()
