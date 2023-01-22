params = {
	'n_data_qubits': 80,
	'n_check_qubits': 70,
	'deg_bit_lower': 2,
	'deg_phase_lower': 2,
	'deg_check_to_check_lower': 2,
	'deg_bit_upper': 10,
	'deg_phase_upper': 10,
	'deg_check_to_check_upper': 10,
	# TODO: delete the below
	'constant_error_rate_lower': 0.0075, # lower bounding here is tricky as we need stat sig
	'constant_error_rate_upper': 0.1,
	'n_genetic_epochs': 10,
	'n_score_testing_samples': 500,
	'n_score_training_per_epoch_initial': 10_000,
	'n_score_training_per_epoch_genetic': 500,
	'n_score_epochs': 1,
	# The number of times to run a decoder to generate the error rate
	'n_decoder_rounds': 50_000,
	'p_skip_mutation': 0.8,
	'p_random_mutation': 0.005
}

# TODO: add in # optimize steps here...
