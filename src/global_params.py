params = {
	'n_data_qubits': 400,
	'n_check_qubits': 192,
	'deg_bit_lower': 3,
	'deg_phase_lower': 3,
	'deg_check_to_check_lower': 2,
	'deg_bit_upper': 8,
	'deg_phase_upper': 8,
	'deg_check_to_check_upper': 8,
	# TODO: delete the below
	'constant_error_rate_lower': 0.002, # lower bounding here is tricky as we need stat sig
	'constant_error_rate_upper': 0.002, #TODO: lets get back to a range but for now we want things to start working...
	'n_genetic_epochs': 10,
	'n_score_testing_samples': 50,
	'n_score_training_per_epoch_initial': 500,#10_000,
	'n_score_training_per_epoch_genetic': 500,
	'n_score_epochs': 5,
	# The number of times to run a decoder to generate the error rate
	'n_decoder_rounds': 500, #TODO: change back to higher
	'p_skip_mutation': 0.8,
	'p_random_mutation': 0.005
}

# TODO: add in # optimize steps here...
