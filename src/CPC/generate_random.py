import CPC.cpc_code as cpc_code
from ldpc_classical import code_gen


def random_cpc(n: int, m_x: int, dv_x: int, m_z: int, dv_z: int, seed_x: int, seed_z:int) -> cpc_code.CPCCode:
    Hx, Hz = code_gen.gen_code(n, m_x, dv_x, seed_x),\
        code_gen.gen_code(n, m_z, dv_z, seed_z)
    return cpc_code.gen_cpc_from_classical_codes(Hx, Hz)
    # return CPCCode(n_bits, n_checks, edges).get_classical_code(), bit_adj, phase_adj, check_check_adj
