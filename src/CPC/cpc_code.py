import numpy as np
import pickle
import numpy.typing as npt


class _BitType():
    def __init__(self) -> None:
        self.BIT_FLIP_DATA = 1
        self.PHASE_FLIP_DATA = 2
        self.PHASE_FLIP_PC = 3


BitType = _BitType()


class CPCVertex:
    def __init__(self, id: int, data_qubit=False, check_qubit=False) -> None:
        if data_qubit and check_qubit:
            raise "Has to be a data or check qubit"
        elif not data_qubit and not check_qubit:
            raise "Have to specify data or check qubit"
        self.data_qubit = data_qubit
        self.check_qubit = check_qubit
        self.id = id

    def __eq__(self, o: object) -> bool:
        return self.data_qubit == o.data_qubit and self.check_qubit == o.check_qubit and self.id == o.id

    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)


class CPCEdge:
    """
        An edge in the CPC graph.
        `bit_check` is only relevant for Check/Data bit edges to indicate whether the edge is a
        Z/X propagation or X/x propagation
    """

    def __init__(self, v1: CPCVertex, v2: CPCVertex, bit_check=True, virtual_edge=False) -> None:
        self.v1 = v1
        self.v2 = v2
        if bit_check and virtual_edge:
            raise ValueError("Cannot have a virtual edge and a bit check")
        if (v1.check_qubit and v2.data_qubit and virtual_edge) or (v1.data_qubit and v2.check_qubit and virtual_edge):
            raise ValueError(
                "Cannot have a virtual edge between check and data qubit")
        if bit_check and (v1.check_qubit and v2.check_qubit):
            raise ValueError("Cannot have same Paulis across check qubits")
        if v1.data_qubit and v2.data_qubit:
            raise ValueError("Cannot have edge between data qubits")

        self.check_to_check = v1.check_qubit and v2.check_qubit
        self.bit_check = bit_check
        self.virtual_edge = virtual_edge

    def __eq__(self, o) -> bool:
        if self.virtual_edge:
            return o.virtual_edge and self.v1 == o.v1 and self.v2 == o.v2 and self.bit_check == o.bit_check
        else:
            return self.bit_check == o.bit_check and ((self.v1 == o.v1 and self.v2 == o.v2) or (self.v1 == o.v2 and self.v2 == o.v1))

    def __ne__(self, __o) -> bool:
        return not self.__eq__(__o)

    def get_opposing_vertex(self, vertex_id) -> CPCVertex:
        return self.v2 if self.v1.id == vertex_id else self.v1


class CPCCode:
    def __init__(self, edges: list[CPCEdge], auto_virtual_edges=False) -> None:
        self.edges = edges

        self.vertex_edge_adj = {}
        vertices = {}

        # Build a vertex adjacency matrix
        for edge in edges:
            if edge.v1.id not in self.vertex_edge_adj:
                self.vertex_edge_adj[edge.v1.id] = []
            self.vertex_edge_adj[edge.v1.id].append(edge)

            if edge.v1.id != edge.v2.id:
                if edge.v2.id not in self.vertex_edge_adj:
                    self.vertex_edge_adj[edge.v2.id] = []
                self.vertex_edge_adj[edge.v2.id].append(
                    edge)

            vertices[edge.v1.id] = edge.v1
            vertices[edge.v2.id] = edge.v2

        self.vertices: list[CPCVertex] = list(vertices.values())

        if auto_virtual_edges:
            print("Simplifying the input code")
            self.simplify_virtual_edges()

    def simplify_virtual_edges(self):
        """
        Simply according to virtual edge rules
        """
        def simp_round():
            for vert in self.vertices:
                simped = self.apply_simplify_rule(vert)
                if simped:
                    # Start the while loop again
                    return True
            return False

        n_simp = 0
        while simp_round():
            n_simp += 1

    def remove_edge(self, edge):
        # TODO: this whole class can be wayyyy more efficient w/ maps instead of lists
        # See https://github.com/Lev-Stambler/AI_QEC/issues/1
        self.vertex_edge_adj[edge.v1.id].remove(edge)
        if edge.v2.id != edge.v1.id:
            self.vertex_edge_adj[edge.v2.id].remove(edge)
        self.edges.remove(edge)

    def get_all_check_vertices(self):
        check_verts = []
        [check_verts.append(v) for v in self.vertices if v.check_qubit]
        return check_verts

    def add_edge(self, edge):
        self.vertex_edge_adj[edge.v1.id].append(edge)
        if edge.v2.id != edge.v1.id:
            self.vertex_edge_adj[edge.v2.id].append(edge)
        self.edges.append(edge)

    def has_edge(self, edge):
        return edge in self.edges

    def get_tanner_graph(self) -> tuple[np.ndarray, np.ndarray]:
        # TODO: somehow enforce PC "types"
        """

        """
        checks: list[CPCVertex] = []
        databits: list[CPCVertex] = []

        for i, v in enumerate(self.vertices):
            if not v.check_qubit and v.data_qubit:
                databits.append(v)
            elif v.check_qubit:
                checks.append(v)

        # As per the paper. https://arxiv.org/pdf/1804.07653.pdf,
        # we have an extra bit for every check qubit
        pc_mat = np.zeros(
            (len(checks), len(checks) + 2 * len(databits)), dtype=np.uint8)

        bit_types = np.zeros(
            len(checks) + 2 * len(databits), dtype=np.uint8)

        checks_to_idx = {}
        for i, check in enumerate(checks):
            checks_to_idx[check.id] = i

        for i, bit in enumerate(databits):
            bit_types[2 * i] = BitType.BIT_FLIP_DATA
            bit_types[2 * i + 1] = BitType.PHASE_FLIP_DATA
            for edge in self.vertex_edge_adj[bit.id]:
                edge: CPCEdge = edge
                if edge.virtual_edge:
                    raise "Virtual edge on databit should be impossible"
                opp_check: CPCVertex = edge.get_opposing_vertex(bit.id)
                if edge.bit_check:
                    assert (opp_check.check_qubit, "Expected check qubit")
                    check_idx = checks_to_idx[opp_check.id]
                    pc_mat[check_idx, 2 * i] = 1
                elif not edge.bit_check:
                    check_idx = checks_to_idx[opp_check.id]
                    pc_mat[check_idx, 2 * i + 1] = 1

        for i, check in enumerate(checks):
            check_bit_idx = 2 * len(databits) + i
            bit_types[check_bit_idx] = BitType.PHASE_FLIP_PC
            for edge in self.vertex_edge_adj[check.id]:
                edge: CPCEdge = edge
                # A virtual edge originating from the check
                if edge.virtual_edge and edge.v1.id == check.id:
                    pc_check_idx = checks_to_idx[edge.v2.id]
                    pc_mat[pc_check_idx, check_bit_idx] = 1
                else:
                    opp_vert = edge.get_opposing_vertex(check.id)
                    if opp_vert.check_qubit:
                        pc_check_idx = checks_to_idx[opp_vert.id]
                        pc_mat[pc_check_idx, check_bit_idx] = 1

        return pc_mat, bit_types

    def apply_simplify_rule(self, vertex: CPCVertex):
        """
        We will have the first rule from https://arxiv.org/pdf/1804.07653.pdf,
        Table II, be searched from the data qubit and the remaining from the check qubit

        Except for the first rule in Table II, we consider cases left to right.
        I.e. the vertex in question is the one on the left side
        """
        if vertex.check_qubit:
            for i in range(len(self.vertex_edge_adj[vertex.id])):
                for j in range(0, i):
                    e1: CPCEdge = self.vertex_edge_adj[vertex.id][i]
                    e2: CPCEdge = self.vertex_edge_adj[vertex.id][j]
                    v1 = e1.v2 if e1.v1.id == vertex.id else e1.v1
                    v2 = e2.v2 if e2.v1.id == vertex.id else e2.v1

                    if v1.id == v2.id:
                        v = v1
                        # Rule 4
                        if v.id == vertex.id and e1.virtual_edge and e2.virtual_edge:
                            self.remove_edge(e1)
                            self.remove_edge(e2)
                            return True
                        # Rule 2
                        elif v.data_qubit and not e1.virtual_edge and not e2.virtual_edge:
                            virtual_edge = None
                            if e1.bit_check and not e2.bit_check:
                                virtual_edge = CPCEdge(
                                    vertex, vertex, bit_check=False, virtual_edge=True)
                            elif not e1.bit_check and e2.bit_check:
                                virtual_edge = CPCEdge(
                                    vertex, vertex, bit_check=False, virtual_edge=True)
                            if virtual_edge is not None and not self.has_edge(virtual_edge):
                                self.add_edge(virtual_edge)
                                return True
                        # Rue 3
                        elif e1.v2.id == vertex.id and e1.virtual_edge and e2.v2.id == vertex.id and e2.virtual_edge:
                            self.remove_edge(e1)
                            self.remove_edge(e2)
                            return True
                        # Rule 5 part 1
                        elif e1.v2.id == vertex.id and e1.virtual_edge and not e2.bit_check and not e2.virtual_edge:
                            virtual_edge = CPCEdge(
                                vertex, v, bit_check=False, virtual_edge=True)
                            # TODO: do we need to check whether things work out here as expected?
                            # I.e. use has_equals
                            # See https://github.com/Lev-Stambler/AI_QEC/issues/1
                            self.remove_edge(e1)
                            self.remove_edge(e2)
                            self.add_edge(virtual_edge)
                            return True
                        # Rule 5 part 2
                        elif e2.v2.id == vertex.id and e2.virtual_edge and not e1.bit_check and not e1.virtual_edge:
                            self.remove_edge(e1)
                            self.remove_edge(e2)
                            virtual_edge = CPCEdge(
                                vertex, v, bit_check=False, virtual_edge=True)
                            self.add_edge(virtual_edge)
                            return True
                        # Rule 6
                        elif (e1.virtual_edge and e2.virtual_edge) and \
                                (
                                    (e1.v1.id == vertex.id and e2.v2.id == vertex.id)
                                    or (e1.v2.id == vertex.id and e2.v1.id == vertex.id)
                        ):
                            self.remove_edge(e1)
                            self.remove_edge(e2)
                            self.add_edge(CPCEdge(vertex, v, bit_check=False))
                            return True

        elif vertex.data_qubit:
            # Rule 1
            for i in range(len(self.vertex_edge_adj[vertex.id])):
                for j in range(0, i):
                    e1: CPCEdge = self.vertex_edge_adj[vertex.id][i]
                    e2: CPCEdge = self.vertex_edge_adj[vertex.id][j]
                    v1 = e1.v2 if e1.v1.id == vertex.id else e1.v1
                    v2 = e2.v2 if e2.v1.id == vertex.id else e2.v1
                    if v1.check_qubit and v2.check_qubit and not e1.virtual_edge and not e2.virtual_edge:
                        virtual_edge = None
                        if e1.bit_check and not e2.bit_check:
                            virtual_edge = CPCEdge(
                                v1, v2, bit_check=False, virtual_edge=True)
                        elif not e1.bit_check and e2.bit_check:
                            virtual_edge = CPCEdge(
                                v2, v1, bit_check=False, virtual_edge=True)
                        if virtual_edge is not None and not self.has_edge(virtual_edge):
                            self.add_edge(virtual_edge)
                            return True
                            # self.vertex_edge_adj[v1].append(virtual_edge)

        return False

    def save(self, filename: str):
        with open(f'{filename}', 'wb') as file:
            pickle.dump(self, file)

    def load(filename: str):
        n: CPCCode = None
        with open(filename, 'rb') as file:
            n = pickle.load(file)
        return n


def gen_cpc_from_classical_codes(H_x: npt.NDArray, H_z: npt.NDArray) -> CPCCode:
    """
    Generate a CPC code from two NP Array Parity Checks. One for H_x (bit flip errors).
    One for H_z (phase flip errors)
    """
    assert (H_x.shape[1] == H_z.shape[1],
            "Expected Hx and Hz to havee the same number of qubits")
    checks_x = [CPCVertex(i, check_qubit=True) for i in range(H_x.shape[0])]
    checks_z = [CPCVertex(i + H_x.shape[0], check_qubit=True)
                for i in range(H_z.shape[0])]
    data_qubits = [CPCVertex(
        i + H_x.shape[0] + H_z.shape[0], data_qubit=True) for i in range(H_x.shape[1])]
    edges = []
    for check_x in range(H_x.shape[0]):
        for qubit in range(H_x.shape[1]):
            if H_x[check_x, qubit]:
                edges.append(CPCEdge(checks_x[check_x], data_qubits[qubit], bit_check=True))

    for check_z in range(H_z.shape[0]):
        for qubit in range(H_z.shape[1]):
            if H_z[check_z, qubit]:
                edges.append(
                    CPCEdge(checks_z[check_z], data_qubits[qubit], bit_check=False))

    return CPCCode(edges, auto_virtual_edges=True)


if __name__ == '__main__':
    code = gen_cpc_from_classical_codes(
        np.array([[1, 0, 1], [0, 1, 1]]), np.array([[1, 0, 1], [0, 1, 1]]))
    print(code.edges)
