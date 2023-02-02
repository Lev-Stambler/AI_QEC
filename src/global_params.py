params = {
    'n_data_qubits': 31,
    'n_check_qubits': 20,
    'deg_bit_lower': 2,
    'deg_phase_lower': 2,
    'deg_check_to_check_lower': 2,
    'deg_bit_upper': 6,
    'deg_phase_upper': 6,
    'deg_check_to_check_upper': 6,
    # lower bounding here is tricky as we need stat sig
    'constant_error_rate_lower': 0.02,
    'constant_error_rate_upper': 0.02,
    'n_genetic_epochs': 10,
    'n_score_testing_samples': 50,
    'n_score_training_per_epoch_initial': 1000,  # 10_000,
    'n_score_training_per_epoch_genetic': 500,
    'n_score_epochs': 5,
    # The number of times to run a decoder to generate the error rate
    'n_decoder_rounds': 100_000,
    'p_skip_mutation': 0.8,
    'p_random_mutation': 0.0005,
    'eval_p_range': [0.001, 0.005, 0.01, 0.015]
}
