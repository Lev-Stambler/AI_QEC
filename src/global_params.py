params = {
	'n_data_qubits': 30,
	'n_check_qubits': 20,
	'deg_bit_lower': 2,
	'deg_phase_lower': 2,
	'deg_check_to_check_lower': 2,
	'deg_bit_upper': 6,
	'deg_phase_upper': 6,
	'deg_check_to_check_upper': 6,
	# TODO: delete the below
	'constant_error_rate_lower': 0.009, # lower bounding here is tricky as we need stat sig
	'constant_error_rate_upper': 0.05,
	'scoring_model_save_path': 'best_scoring_model',
	'n_genetic_epochs': 3,
	'n_score_testing_samples': 500,
	'n_score_training_per_epoch_initial': 1_000,
	'n_score_training_per_epoch_genetic': 400,
	'n_score_epochs': 2,
	# The number of times to run a decoder to generate the error rate
	'n_decoder_rounds': 10_000
}

# TODO: add in # optimize steps here...
