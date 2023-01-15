from CPC import generate_random

def generate_code(params):
	return generate_random.random_cpc(params['n_data_qubits'], params['n_check_qubits'], params['deg_phase'], params['deg_bit'], params['deg_check_to_check'])