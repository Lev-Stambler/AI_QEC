from hashlib import sha1
from json import JSONEncoder
import json
import random
import numpy as np
import numpy.typing as npt
import torch

from CPC import cpc_code


def sign_to_bin(x):
    return 0.5 * (1 - x)


def bin_to_sign(x):
    return 1 - 2 * x


def EbN0_to_std(EbN0, rate):
    snr = EbN0 + 10. * np.log10(2 * rate)
    return np.sqrt(1. / (10. ** (snr / 10.)))


def BER(x_pred, x_gt):
    return torch.mean((x_pred != x_gt).float()).item()


def FER(x_pred, x_gt):
    return torch.mean(torch.any(x_pred != x_gt, dim=1).float()).item()

# Thanks to https://github.com/christiansiegel/coding-theory-algorithms/blob/master/LinearBlockCode.py


def GtoP(G):
    """Extract the submatrix P from a Generator Matrix in systematic form.
    Args:
        G: Generator Matrix in systematic form
    Returns:
        Submatrix P of G.
    """
    k = G.shape[0]
    n = G.shape[1]
    P = G[:k, :n - k]
    return P


def HtoSystematicH(H):
    """Convert a parity check matrix into systematic form,
    gotten from https://gist.github.com/popcornell/bc29d1b7ba37d824335ab7b6280f7fec"""
    m, n = H.shape

    i = 0
    j = 0

    while i < m and j < n:
        # find value and index of largest element in remainder of column j
        k = np.argmax(H[i:, j]) + i

        # swap rows
        # M[[k, i]] = M[[i, k]] this doesn't work with numba
        temp = np.copy(H[k])
        H[k] = H[i]
        H[i] = temp

        aijn = H[i, j:]

        # make a copy otherwise M will be directly affected
        col = np.copy(H[:, j])

        col[i] = 0  # avoid xoring pivot row with itself

        flip = np.outer(col, aijn)

        H[:, j:] = H[:, j:] ^ flip

        i += 1
        j += 1

    return H


def HtoP(H):
    """Extract the submatrix P from a Parity Check Matrix in systematic form.
    Args:
        H: Parity Check Matrix in systematic form
    Returns:
        Submatrix P of G.
    """
    n = H.shape[1]
    k = n - H.shape[0]
    PK = H[:, n - k:n]
    P = PK.transpose(0, 1)
    return P


def HtoG(H):
    """Convert a Parity Check Matrix in systematic form to a Generator Matrix.
    Args:
        H: Parity Check Matrix in systematic form
    Returns:
        Generator Matrix G
    """
    n = H.shape[1]
    k = n - H.shape[0]
    P = HtoP(H)
    Ik = np.eye(k)
    G = np.concatenate((P, Ik), axis=0)
    return G


# def sample_iid_error(n):
#     p = random.random() * (params['constant_error_rate_upper'] - params['constant_error_rate_lower']) + params['constant_error_rate_lower']
#     return np.ones((n)) * p


def get_numb_type():
    return torch.float32 if torch.cuda.is_available() else torch.double


def numpy_to_parameter(np_arr, device, unsqueeze=True):
    if unsqueeze:
        return torch.nn.parameter.Parameter(
            torch.from_numpy(np_arr).to(device).type(get_numb_type()).unsqueeze(0))
    else:
        return torch.nn.parameter.Parameter(
            torch.from_numpy(np_arr).to(device).type(get_numb_type()))


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# def get_most_recent_model_path_rl_info():
#     return f"{get_most_recent_model_path_rl()}.json"

# def get_most_recent_model_path_rl():
#     return f"best_scoring/best_model_rl_{params['params_prefix']}_({params['n_data_qubits']},{params['n_data_qubits'] - params['n_check_qubits']}).zip_2"

# def get_best_scoring_model_path():
#     return f"best_scoring/best_model_{params['params_prefix']}_({params['n_data_qubits']},{params['n_data_qubits'] - params['n_check_qubits']})"

# def get_eval_path():
#     return f"eval/results_{params['params_prefix']}_({params['n_data_qubits']},{params['n_data_qubits'] - params['n_check_qubits']}).json"

# def get_eval_baseline_path():
#     return f"eval/results_baseline_{params['params_prefix']}_({params['n_data_qubits']},{params['n_data_qubits'] - params['n_check_qubits']}).json"


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj * 1_000)
        if isinstance(obj, np.floating):
            return float(obj * 1_000)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class TrialParam():
    def get_key(self):
        return sha1(json.dumps(vars(self), sort_keys=True).encode('utf-8')).hexdigest()


class CPCTrialParams(TrialParam):
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
