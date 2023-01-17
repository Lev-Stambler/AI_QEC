params = {
	'n_data_qubits': 30,
	'n_check_qubits': 20,
	'deg_bit': 4,
	'deg_phase': 4,
	'deg_check_to_check': 2,
	# TODO: delete the below
	'constant_error_rate': 0.01,
	'scoring_model_save_path': 'best_scoring_model',
	'n_genetic_epochs': 3,
	'n_score_testing_samples': 500,
	'n_score_training_samples_initial': 2_000,
	'n_score_training_samples_genetic': 1_000,
	'n_score_epochs': 1,
	# The number of times to run a decoder to generate the error rate
	'n_decoder_rounds': 10_000
}

# TODO: add in # optimize steps here...
