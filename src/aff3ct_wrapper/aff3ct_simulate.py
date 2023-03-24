import sys
import os

import numpy as np
import math
import numpy.typing as npt
from CPC import generate_random

_TMP_ALIST_PATH = "build/tmp_code.alist"
_TMP_OUT_PATH = "build/tmp_out.txt"


def _np_to_alist(H: npt.NDArray, alist_path=_TMP_ALIST_PATH):
    H = H.astype(np.uint16)
    n_pc = H.shape[0]
    n_bits = H.shape[1]

    pc_degs = [np.sum(H[i]) for i in range(n_pc)]
    bit_degs = [np.sum(H[:, i]) for i in range(n_bits)]
    max_pc_deg = int(max(pc_degs))
    max_bit_deg = int(max(bit_degs))

    def row_to_alist(row: npt.NDArray, size):
        adj = []
        for i, b in enumerate(row):
            if int(b) == 1:
                adj.append(str(i + 1))  # offset by 1
        q = len(adj)
        for i in range(q, size):
            adj.append(str(0))
        return " ".join(adj)

    alist = f"{n_bits} {n_pc}" + "\n" + \
        f"{max_bit_deg} {max_pc_deg}" + "\n" + \
        ' '.join([str(int(p)) for p in bit_degs]) + "\n" + \
        ' '.join([str(int(p)) for p in pc_degs]) + "\n" + \
        '\n'.join([row_to_alist(H[:, i], max_bit_deg) for i in range(n_bits)]) + "\n" + \
        '\n'.join([row_to_alist(H[i], max_pc_deg) for i in range(n_pc)]) + "\n"
    
    f = open(alist_path, "w")
    f.write(alist)
    f.close()

def parse_out_get_fer(path=_TMP_OUT_PATH):
    f = open(path)
    s = f.read()
    f.close()

    spls = s.split('\n')
    info_line = spls[-3]
    relevant_sect = info_line.split('||')[1]
    fer = float(relevant_sect.split('|')[-1].strip())
    return fer


# TODO: support more than just BSC nois01ee
# See https://github.com/Lev-Stambler/AI_QEC/issues/3
# TODO: support parameter setting from a different file/ object
# TODO: its worth trying to write out n_frame_err but the lower the error rate, the smaller our std bars as is...
# That being said, working something exp out would be nices
def get_wsr(H: npt.NDArray, err_distr: npt.NDArray, n_frame_errors=1_000, err_bar_cutoff=0.01): # TODO: param n_frame_errors...
    _np_to_alist(H)

    channel_type = "BSC"
    n = H.shape[-1]
    k_pc = n - H.shape[0]
    # TODO: BP flooding is the default but its worth looking into alternatives
    # See the options: https://aff3ct.readthedocs.io/en/latest/user/simulation/parameters/codec/ldpc/decoder.html
    dec_type = "BP_FLOODING"
    # Maybe we want to use
    dec_implem = "AMS"  # hmm... use MS (min sum)
    dec_bp_iterations = 10  # Again, we are using the default here
    sim_noise_type = "EP" # Event probability
    # for BSC, OOK modulation is required. See https://aff3ct.readthedocs.io/en/latest/user/simulation/parameters/channel/channel.html#chn-chn-type
    run_sim_cmd = f"aff3ct --sim-cde-type LDPC --chn-type {channel_type} --enc-cw-size {n} --enc-info-bits {k_pc} " \
        + f"--enc-type LDPC_H --dec-h-path {_TMP_ALIST_PATH} --dec-type {dec_type} --dec-implem {dec_implem} " + \
        f"--dec-ite {dec_bp_iterations} --sim-noise-type {sim_noise_type} --sim-noise-range '{','.join([str(p) for p in err_distr])}' --mdm-type OOK " + \
        f"--mnt-max-fe {n_frame_errors} > {_TMP_OUT_PATH}"

    os.system(run_sim_cmd)
    return 1 - parse_out_get_fer()
