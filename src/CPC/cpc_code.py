import numpy.typing as npt


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
        `opposing_pauli` is only relevant for Check/Data bit edges to indicate whether the edge is a
        Z/X propagation or X/x propagation
    """

    def __init__(self, v1: CPCVertex, v2: CPCVertex, opposing_pauli=True, virtual_edge=False) -> None:
        self.v1 = v1
        self.v2 = v2
        if not opposing_pauli and virtual_edge:
            raise "Cannot have a virtual edge and opposing Pauli"
        if (v1.check_qubit and v2.data_qubit and virtual_edge) or (v1.data_qubit and v2.check_qubit and virtual_edge):
            raise "Cannot have a virtual edge between check and data qubit"
        if not opposing_pauli and (v1.check_qubit and v2.check_qubit):
            raise "Cannot have same Paulis across check qubits"
        if v1.data_qubit and v2.data_qubit:
            raise "Cannot have edge between data qubits"

        self.opposing_pauli = opposing_pauli
        self.virtual_edge = virtual_edge

    def __eq__(self, o) -> bool:
        if self.virtual_edge:
            return o.virtual_edge and self.v1 == o.v1 and self.v2 == o.v2 and self.opposing_pauli == o.opposing_pauli
        else:
            return self.opposing_pauli == o.opposing_pauli and ((self.v1 == o.v1 and self.v2 == o.v2) or (self.v1 == o.v2 and self.v2 == o.v1))

    def __ne__(self, __o) -> bool:
        return not self.__eq__(__o)


class CPCCode:
    def __init__(self, edges: list[CPCEdge], auto_simplify=True) -> None:
        self.edges = edges

        self.vertex_edge_adj = {}
        vertices = {}

        # Build a vertex adjacency matrix
        for edge in edges:
            if edge.v1.id not in self.vertex_edge_adj:
                self.vertex_edge_adj[edge.v1.id] = []
            self.vertex_edge_adj[edge.v1.id].append(edge)

            if edge.v2.id not in self.vertex_edge_adj:
                self.vertex_edge_adj[edge.v2.id] = []
            self.vertex_edge_adj[edge.v2.id].append(
                edge)  # TODO: do we want this here?

            vertices[edge.v1.id] = edge.v1
            vertices[edge.v2.id] = edge.v2

        self.vertices = vertices.values()

        if auto_simplify:
            self.simplify()

    def simplify(self):
        """
        Simply according to virtual edge rules
        """
        all_simplified = False
        n_simp = 0
        while not all_simplified:
            all_simplified = True
            for vert in self.vertices:
                simped = self.apply_simplify_rule(vert)
                if simped:
                    # Start the while loop again
                    all_simplified = False
                    print("SIMPED", n_simp, len(self.edges), len(self.vertices))
                    n_simp += 1
                    break

    def get_classical_codes(self) -> list[npt.NDArray]:
        """
            Return the parity check matrix associated with the underlying
            classical codes
        """
        pass

    def remove_edge(self, edge):
        # TODO: this whole class can be wayyyy more efficient w/ maps instead of lists
        self.vertex_edge_adj[edge.v1.id].remove(edge)
        if edge.v2.id != edge.v1.id:
            self.vertex_edge_adj[edge.v2.id].remove(edge)
        self.edges.remove(edge)

    def add_edge(self, edge):
        self.vertex_edge_adj[edge.v1.id].append(edge)
        if edge.v2.id != edge.v1.id:
            self.vertex_edge_adj[edge.v2.id].append(edge)
        self.edges.append(edge)
    
    def has_edge(self, edge):
        return edge in self.edges

    def apply_simplify_rule(self, vertex: CPCVertex):
        """
        We will have the first rule from https://arxiv.org/pdf/1804.07653.pdf,
        Table II, be searched from the data qubit and the remaining from the check qubit

        Except for the first rule in Table II, we consider cases left to right.
        I.e. the vertex in question is the one on the left side
        """
        if vertex.data_qubit:
            # Rule 1
            for i in range(len(self.vertex_edge_adj[vertex.id])):
                for j in range(0, i):
                    e1: CPCEdge = self.vertex_edge_adj[vertex.id][i]
                    e2: CPCEdge = self.vertex_edge_adj[vertex.id][j]
                    v1 = e1.v2 if e1.v1.id == vertex.id else e1.v1
                    v2 = e2.v2 if e2.v1.id == vertex.id else e2.v1
                    if v1.check_qubit and v2.check_qubit and not e1.virtual_edge and not e2.virtual_edge:
                        virtual_edge = None
                        if e1.opposing_pauli and not e2.opposing_pauli:
                            virtual_edge = CPCEdge(
                                v1, v2, opposing_pauli=True, virtual_edge=True)
                        elif not e1.opposing_pauli and e2.opposing_pauli:
                            virtual_edge = CPCEdge(
                                v2, v1, opposing_pauli=True, virtual_edge=True)
                        if virtual_edge is not None and not self.has_edge(virtual_edge):
                            self.add_edge(virtual_edge)
                            return True
                            # self.vertex_edge_adj[v1].append(virtual_edge)
                            # self.vertex_edge_adj[v2].append(virtual_edge)
                            # self.edges.append(virtual_edge)
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
                        elif v.data_qubit:
                            virtual_edge = None
                            if e1.opposing_pauli and not e2.opposing_pauli:
                                virtual_edge = CPCEdge(
                                    vertex, vertex, virtual_edge=True)
                            elif not e1.opposing_pauli and e2.opposing_pauli:
                                virtual_edge = CPCEdge(
                                    vertex, vertex, virtual_edge=True)
                            if virtual_edge is not None and not self.has_edge(virtual_edge):
                                self.add_edge(virtual_edge)
                                return True
                        # Rue 3
                        elif e1.v2.id == vertex.id and e1.virtual_edge and e2.v2.id == vertex.id and e2.virtual_edge:
                            self.remove_edge(e1)
                            self.remove_edge(e2)
                            return True
                        # Rule 5 part 1
                        elif e1.v2.id == vertex.id and e1.virtual_edge and e2.opposing_pauli and not e2.virtual_edge:
                            virtual_edge = CPCEdge(vertex, v, virtual_edge=True)
                            # TODO: do we need to check whether things work out here as expected?
                            # I.e. use has_equals
                            self.remove_edge(e1)
                            self.remove_edge(e2)
                            self.add_edge(virtual_edge)
                            return True
                        # Rule 5 part 2
                        elif e2.v2.id == vertex.id and e2.virtual_edge and e1.opposing_pauli and not e1.virtual_edge:
                            self.remove_edge(e1)
                            self.remove_edge(e2)
                            virtual_edge = CPCEdge(vertex, v, virtual_edge=True)
                            self.add_edge(virtual_edge)
                            return True
                        # Rule 6
                        elif (e1.virtual_edge and e2.virtual_edge) and ((e1.v1.id == vertex.id and e2.v2.id == vertex.id) or (e1.v2.id == vertex.id and e2.v1.id == vertex.id)):
                            self.remove_edge(e1)
                            self.remove_edge(e2)
                            self.add_edge(CPCEdge(vertex, v, opposing_pauli=True))
                            return True

        return False
