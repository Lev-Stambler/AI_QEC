import numpy as np
import pickle
import numpy.typing as npt


class _BitType():
    def __init__(self) -> None:
        self.BIT_FLIP_DATA = 1
        self.PHASE_FLIP_DATA = 2
        self.PHASE_FLIP_PC = 3


BitType = _BitType()


def get_cpc_parity_mat(bit_mat, phase_mat, check_mat, concat_fn=np.concatenate, eye_fn=np.eye):
    H = np.concatenate([phase_mat, (bit_mat.T @ phase_mat.T)
                       ^ check_mat, eye_fn(phase_mat.shape[0])], dim=-1)
    print("AAAA", H.shape)
    return H
