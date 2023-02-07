import numpy as np
import stim
import numpy.typing as npt

from noise import NoisePauli


def zero_noisy_codeword_circuit(H: npt.NDArray, p_meas: NoisePauli):
	assert H.shape[1] % 2 == 0, "Quantum parity check matrix must have an even width"
	n_stabilizers = H.shape[0]
	n_qubits = H.shape[1] // 2

	circuit = stim.Circuit()

	# Build the circuit to get a codeword
	for i in range(n_stabilizers):
		for j in range(n_qubits):
			if H[i, n_qubits + j] and H[i, j]:
				circuit.append(f"PAULI_CHANNEL_1({p_meas.p_y})", j)
				circuit.append("MY", j)
			elif H[i, j]:
				circuit.append(f"PAULI_CHANNEL_1({p_meas.p_x})", j)
				circuit.append("MX", j)
			elif H[i, n_qubits + j]:
				circuit.append(f"PAULI_CHANNEL_1({p_meas.p_z})", j)
				circuit.append("MZ", j)

	# Let's see the circuit's representation using stim's circuit language:
	print("Circuit representation", repr(circuit))
	return circuit
