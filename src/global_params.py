params = {
    'n_data_qubits': 28,
    'n_check_qubits': 20,
    'deg_bit_lower': 3,
    'deg_phase_lower': 3,
    'deg_check_to_check_lower': 2,
    'deg_bit_upper': 6,
    'deg_phase_upper': 6,
    'deg_check_to_check_upper': 6,
    # lower bounding here is tricky as we need stat sig
    # TODO: turn into a range instead of 2 variables
    'constant_error_rate_lower': 0.002,
    'constant_error_rate_upper': 0.002,
    'n_genetic_epochs': 20,
    'n_score_testing_samples': 50,
    'n_score_training_per_epoch_initial': 10_000,  # 10_000,
    'n_score_training_per_epoch_genetic': 4_000,
    'n_score_epochs_initial': 3,
    'n_score_epochs_genetic': 2,
    # The number of times to run a decoder to generate the error rate
    'n_decoder_rounds': 100_000,
    'p_skip_mutation': 0.35,
    'p_random_mutation_range': [0.005, 0.015],
    'eval_p_range': [0.001, 0.005],
    'params_prefix': 'bsc_iid_noise',
    ########## RL parameters ####################
    'rl_save_model_freq': 1_000
}
