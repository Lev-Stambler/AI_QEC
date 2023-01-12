from CPC import generate_random
from global_params import params

def generate_code():
	return generate_random.random_cpc(params['n_data_qubits'], params['n_qubit_checks'], params['deg_phase'], params['deg_bit'], params['deg_cc']).get_classical_code()