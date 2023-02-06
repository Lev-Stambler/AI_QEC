import numpy.typing as npt
import os

def np_to_alist(H: npt.NDArray, alist_path):
	pass

# TODO: support more than just BSC noise
# See https://github.com/Lev-Stambler/AI_QEC/issues/3
# TODO: support parameter setting from a different file/ object
def get_wer(H: npt.NDArray, err_distr: npt.NDArray, n_max_trials=1_000_000, err_bar_cutoff=0.01):
	alist_path = "TODO::::"
	np_to_alist(H, alist_path)

	channel_type = "BSC"
	n = H.shape[-1]
	k_pc = n - H.shape[0]
	# TODO: BP flooding is the default but its worth looking into alternatives
	# See the options: https://aff3ct.readthedocs.io/en/latest/user/simulation/parameters/codec/ldpc/decoder.html
	dec_type = "BP_FLOODING"
	# Maybe we want to use
	dec_implem = "AMS" # hmm... use MS (min sum)
	dec_bp_iterations = 10 # Again, we are using the default here
	run_sim_cmd = f"aff3ct --chn-type {channel_type} --enc-cw-size {n} --enc-info-bits {k_pc} " \
			+ f"--enc-type LDPC_H --dec-h-path {alist_path} --dec-type {dec_type} --dec-implem {dec_implem} " + \
				f"--dec-ite {dec_bp_iterations} "
	print(run_sim_cmd)
	os.system(run_sim_cmd)
	pass

