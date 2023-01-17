params = {
	'n_data_qubits': 90,
	'n_check_qubits': 70,
	'deg_bit_lower': 2,
	'deg_phase_lower': 2,
	'deg_check_to_check_lower': 2,
	'deg_bit_upper': 8,
	'deg_phase_upper': 8,
	'deg_check_to_check_upper': 8,
	# TODO: delete the below
	'constant_error_rate_lower': 0.05, # lower bounding here is tricky as we need stat sig
	'constant_error_rate_upper': 0.1,
	'scoring_model_save_path': 'best_scoring_model',
	'n_genetic_epochs': 10,
	'n_score_testing_samples': 500,
	'n_score_training_per_epoch_initial': 2_000,
	'n_score_training_per_epoch_genetic': 400,
	'n_score_epochs': 1,
	# The number of times to run a decoder to generate the error rate
	'n_decoder_rounds': 50_000
}

# TODO: add in # optimize steps here...
