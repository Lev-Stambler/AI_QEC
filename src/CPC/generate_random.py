import numpy as np

from CPC.cpc_code import CPCCode, CPCVertex, CPCEdge


def random_cpc(n_bits: int, n_checks: int, deg_vc_opp: int, deg_vc_same: int, deg_cc: int) -> CPCCode:
    bit_vertices = [CPCVertex(i, data_qubit=True) for i in range(n_bits)]
    check_vertices = [CPCVertex(i + n_bits, check_qubit=True) for i in range(n_checks)]
    edges = []
    for bit_vert in bit_vertices:
        # TODO: this may be quite inefficient
        check_idx = np.random.permutation(n_checks)[:deg_vc_opp]
        for c in check_idx:
            edges.append(CPCEdge(bit_vert, check_vertices[c], bit_check=True))

        check_idx = np.random.permutation(n_checks)[:deg_vc_same]
        for c in check_idx:
            edges.append(CPCEdge(bit_vert, check_vertices[c], bit_check=False))
    for check_vert in check_vertices:
        check_idx = np.random.permutation(n_checks)[:deg_cc]
        for c in check_idx:
            edges.append(CPCEdge(check_vert, check_vertices[c]))

    return CPCCode(n_bits, n_checks, edges)
        
