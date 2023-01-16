import numpy as np

from CPC.cpc_code import CPCCode, CPCVertex, CPCEdge


def random_cpc(n_bits: int, n_checks: int, deg_phase: int, deg_bit: int, deg_cc: int) -> CPCCode:
    """
    TODO: I think "relaxing" the degree requirement would make life **a lot**
    TODO: easier when generalizing the search space. Maybe we have like a "range"
    TODO: and uniformly sample from there. Selection for lower degree or not can be built into
    TODO: the error simulator via Stim
    """
    bit_vertices = [CPCVertex(i, data_qubit=True) for i in range(n_bits)]
    check_vertices = [CPCVertex(i + n_bits, check_qubit=True) for i in range(n_checks)]
    edges = []
    bit_adj = np.zeros((n_bits, n_checks), dtype=np.int16)
    phase_adj = np.zeros((n_bits, n_checks), dtype=np.int16)
    check_check_adj = np.zeros((n_checks, n_checks), dtype=np.int16)

    for i, bit_vert in enumerate(bit_vertices):
        # TODO: this may be quite inefficient
        check_idx = np.random.permutation(n_checks)[:deg_phase]
        for c in check_idx:
            edges.append(CPCEdge(bit_vert, check_vertices[c], bit_check=True))
            bit_adj[i, c] = 1

        check_idx = np.random.permutation(n_checks)[:deg_bit]
        for c in check_idx:
            edges.append(CPCEdge(bit_vert, check_vertices[c], bit_check=False))
            phase_adj[i, c] = 1

    for c1, check_vert in enumerate(check_vertices):
        check_idx = np.random.permutation(n_checks)[:deg_cc]
        for c2 in check_idx:
            edges.append(CPCEdge(check_vert, check_vertices[c2]))
            check_check_adj[c1, c2] = 1
            check_check_adj[c2, c1] = 1

    return CPCCode(n_bits, n_checks, edges).get_classical_code(), bit_adj, phase_adj, check_check_adj
        
