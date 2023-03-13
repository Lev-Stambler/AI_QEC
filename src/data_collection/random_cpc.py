from hashlib import sha1
import os
from CPC import cpc_code
import json
from ldpc_classical import code_gen
import random
from ldpc_classical.aff3ct_wrapper import aff3ct_simulate
from scoring.score_dataset import run_decoder_bp_osd
import numpy as np
import numpy.typing as npt


class TrialParam():
    def get_key(self):
        return sha1(json.dumps(vars(self), sort_keys=True).encode('utf-8')).hexdigest()


class RandomCPCTrialParams(TrialParam):
    """
    A class for Random CPC trial parameters
    """

    def __init__(
            self, n: int, m_x: int, dv_x: int, m_z: int, dv_z: int, n_cross_checks: int,
            p_error: list[float], seeds: list[int]
    ) -> None:
        """
        Parameters
        -----------
        n: The number of bits in the classical codes
        m_x: The number of X (bit) type parity checks
        dv_x: The degree of the qubits for the subgraph connected to X type parity checks
        m_z: The number of Z (phase) type parity checks
        dv_z: The degree of the qubits for the subgraph connected to Z type parity checks
        n_cross_checks: The number of cross checks
        p_error: a list of X, Y, and Z errors respectively
        seeds: a list of length two to be used as seeds for the classical codes
        """
        self.type = "Random Trial"
        self.n = n
        self.m_x = m_x
        self.m_z = m_z
        self.dv_x = dv_x
        self.dv_z = dv_z
        self.n_cross_checks = n_cross_checks
        self.p_error = p_error
        self.seeds = seeds


def _add_random_cross_checks(cpc: cpc_code.CPCCode, n_cross_checks):
    i = 0
    check_verts = []
    [check_verts.append(v) for v in cpc.vertices if v.check_qubit]
    sample_list = list(range(len(check_verts)))

    while i < n_cross_checks:
        idxs = random.sample(sample_list, 2)
        cross_edge = cpc_code.CPCEdge(
            check_verts[idxs[0]], check_verts[idxs[1]], bit_check=False)
        if not cpc.has_edge(cross_edge):
            cpc.add_edge(cross_edge)
            i += 1

# TODO: move elsewhere


def calculate_tanner_p_error_depolarizing(bit_types: npt.NDArray, px: float, py: float, pz: float):
    p_errors = np.zeros((bit_types.shape))
    for i in range(bit_types.shape[-1]):
        if bit_types[i] == cpc_code.BitType.BIT_FLIP_DATA:
            p_errors[i] = px * (1 - py) + py * (1 - px) - px * py
        elif bit_types[i] == cpc_code.BitType.PHASE_FLIP_DATA or bit_types[i] == cpc_code.BitType.PHASE_FLIP_PC:
            p_errors[i] = pz * (1 - py) + py * (1 - pz) - pz * py
        else:
            ValueError("Unexpected bit type")
    return p_errors

# TODO: saving to file stuff


def save_wer(file: str, param: RandomCPCTrialParams, wer):
    wers = {}
    if os.path.isfile(file):
        f = open(file, "r")
        f_content = f.read()
        wers = json.loads(f_content)
        f.close()

    k = param.get_key()
    wers[k] = {}
    wers[k]["wer"] = wer
    wers[k]["params"] = vars(param)
    f = open(file, "w")
    f.write(json.dumps(wers))
    f.close()

# TODO: memo with the codes?


def random_code_trials(params: list[RandomCPCTrialParams], cpc_code_save_dir: str, save_res_file: str):
    out = []
    for param in params:
        cpc_path = f"{cpc_code_save_dir}/{param.get_key()}.cpc"
        cpc = None
        if not os.path.exists(cpc_path):
            Hx, Hz = code_gen.gen_code(param.n, param.m_x, param.dv_x, param.seeds[0]),\
                code_gen.gen_code(param.n, param.m_z, param.dv_z, param.seeds[1])
            cpc = cpc_code.gen_cpc_from_classical_codes(Hx, Hz)
            cpc.save(cpc_path)
        else:
            print("Loading")
            cpc = cpc_code.CPCCode.load(cpc_path)
        _add_random_cross_checks(cpc, param.n_cross_checks)
        H, bit_types = cpc.get_tanner_graph()
        err = calculate_tanner_p_error_depolarizing(
            bit_types, param.p_error[0], param.p_error[1], param.p_error[2])
        wer = aff3ct_simulate.get_wer(H, err)
        out.append(wer)
        save_wer(save_res_file, param, wer)
        # cpc_code.BitType.
