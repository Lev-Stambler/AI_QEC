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


def initialize_scoring_model(device, plot_loss=None, skip_testing=False, scoring_model=None, initialize_epoch_start=1):
    def gc(): return scoring.initial_code_sampling.generate_code()
    sample_code, _, _, _ = gc()
    n = sample_code.shape[-1]
    k = n - sample_code.shape[-2]
    def ge(): return utils.sample_iid_error(n)
    N_dec = 6 # CHanged from 6
    h = 8  # changed from 8
    d_model = 32
    model = score_model.ScoringTransformer(
        params['n_data_qubits'], params['n_check_qubits'], h, d_model, N_dec, device, dropout=0).to(device) if scoring_model is None else scoring_model
    scoring.score_training.main_training_loop(
        "initialization", model, ge, gc, utils.get_best_scoring_model_path(), params['n_score_training_per_epoch_initial'], plot_loss, skip_testing=skip_testing, epoch_start=initialize_epoch_start)
    return model
    # model = torch.load(os.path.join(save_path, 'best_model'))


def evaluate_performance(scoring_model: score_model.ScoringTransformer, gen_model: gen_model.GeneratingModel, p_phys_flips: list[int], low_p: list[int], epoch, n_tests=5_000, averaging_samples=5, out_file='results.json'):
    n = (params['n_data_qubits'] + params['n_check_qubits']) * 2

    json_object = None
    print(f"Starting to evaluate performance for epoch {epoch}")
    with open(out_file, 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
    json_object[f"epoch_{epoch}"] = {}
    best_low_p_succ_rate = [0.0] * len(low_p)
    for p in p_phys_flips:
        cum_succ_rate = 0
        for _ in range(averaging_samples):
            err = np.ones(n) * p
            pc, _, _, _ = gen_model.generate_sample(
                scoring_model, err, mutate=False)
            succ_rate = run_decoder(pc, n_tests, err, multiproc=False)
            cum_ucc_rate += succ_rate
            for i in range(len(low_p)):
                p = low_p[i]
                err = np.ones(n) * p
                pc, _, _, _ = gen_model.generate_sample(
                    scoring_model, err, mutate=False)
                r = run_decoder(pc, n_tests, err, multiproc=False)
                if r > best_low_p_succ_rate[i]:
                    best_low_p_succ_rate[i] = r
        json_object[f"epoch_{epoch}"][f"p_{p}"] = cum_succ_rate / averaging_samples

        for p, best_wsr in zip(low_p, best_low_p_succ_rate):
            json_object[f"epoch_{epoch}"][f"low_p_best_{p}"] = best_wsr

    with open(out_file, "w") as outfile:
        json.dump(json_object, outfile)
    print(f"Done evaluating performance for epoch {epoch}")


def train_score_model_with_generator(genetic_epoch, scoring_model: score_model.ScoringTransformer, scoring_model_copy: score_model.ScoringTransformer,
                                     generating_model: gen_model.GeneratingModel, plot_loss=None, skip_testing=False):
    n = (params['n_data_qubits'] + params['n_check_qubits']) * 2

    def ge(): return utils.sample_iid_error(n)
    err = ge()
    def gc(): return generating_model.generate_sample(scoring_model_copy, err)

    scoring.score_training.main_training_loop(f"gen_epoch_{genetic_epoch}",
        scoring_model, ge, gc, utils.get_best_scoring_model_path(), params['n_score_training_per_epoch_genetic'],  plot_loss, skip_testing=skip_testing)


def main(plot_loss=None, load_saved_scoring_model=False, load_saved_generating_model=False, skip_testing=False, skip_initialization_training=False, skip_eval=False, initialize_epoch_start=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(utils.get_numb_type())
    scoring_model = None
    if load_saved_scoring_model:
        scoring_model = torch.load(os.path.join(
            utils.get_best_scoring_model_path()))

    if not skip_initialization_training:
        scoring_model = initialize_scoring_model(
            device, plot_loss, skip_testing=skip_testing, scoring_model=scoring_model, initialize_epoch_start=initialize_epoch_start)
    if not load_saved_generating_model:
        generating_model = gen_model.GeneratingModel(
            device=device,
            p_skip_mutation=params['p_skip_mutation'],
            p_random_mutation=params['p_random_mutation'],
        )
    else:
        # TODO: load generating model??
        pass

    p_eval_range = [params['constant_error_rate_lower'],
               params['constant_error_rate_upper']]
    for genetic_epoch in range(params['n_genetic_epochs']):
        if not skip_eval:
            evaluate_performance(scoring_model, generating_model, p_eval_range, [
                                 0.001, 0.005, 0.01, 0.015], genetic_epoch)
        print(f"Starting epoch #{genetic_epoch + 1}")
        scoring_copied = copy.deepcopy(scoring_model)
        train_score_model_with_generator(genetic_epoch,
            scoring_model, scoring_copied, generating_model, plot_loss, skip_testing=skip_testing)
        del scoring_copied
        # TODO: make p_range better
    if not skip_eval:
        evaluate_performance(scoring_model, generating_model, p_eval_range, [
                            0.001, 0.005, 0.01, 0.015], genetic_epoch)

    return scoring_model


if __name__ == '__main__':
    main()
