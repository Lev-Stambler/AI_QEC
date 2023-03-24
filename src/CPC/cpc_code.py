import numpy as np
import numpy.typing as npt

def get_classical_code_cpc(bit_adj, phase_adj, check_adj) -> npt.NDArray:
    """
        Return the parity check matrix associated with the underlying
        classical codes
    """
            # The yellow nodes in the paper
    pc = np.concatenate([phase_adj.transpose(), (((bit_adj.transpose() @ phase_adj) % 2) + check_adj) % 2, bit_adj.transpose(), np.eye(check_adj.shape[-1])], axis=-1)
    return np.array(pc, dtype=np.uint8)