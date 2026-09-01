"""
Quantum Engine for AI-Driven Predictive Healthcare Analytics
Implements Quantum Circuit Simulations, Bell State Encodings (Table 5),
and Figure 8 3-Qubit Quantum Circuit directly based on Base1.pdf.
"""

import numpy as np
from typing import Dict, List, Any, Tuple

# Base Quantum Gates
I2 = np.array([[1, 0], [0, 1]], dtype=complex)
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

def unitary_gate(theta: float, phi: float, lam: float) -> np.ndarray:
    """
    Standard single-qubit parameter rotation gate U(theta, phi, lambda) as in Fig. 8:
    U(theta, phi, lambda) = [
        [cos(theta/2), -exp(i*lam)*sin(theta/2)],
        [exp(i*phi)*sin(theta/2), exp(i*(phi+lam))*cos(theta/2)]
    ]
    """
    return np.array([
        [np.cos(theta / 2.0), -np.exp(1j * lam) * np.sin(theta / 2.0)],
        [np.exp(1j * phi) * np.sin(theta / 2.0), np.exp(1j * (phi + lam)) * np.cos(theta / 2.0)]
    ], dtype=complex)

def cnot_gate(n_qubits: int, control: int, target: int) -> np.ndarray:
    """Constructs an n-qubit CNOT gate matrix."""
    dim = 2 ** n_qubits
    op = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        bit_ctrl = (i >> (n_qubits - 1 - control)) & 1
        if bit_ctrl == 1:
            j = i ^ (1 << (n_qubits - 1 - target))
            op[j, i] = 1.0
        else:
            op[i, i] = 1.0
    return op

def single_qubit_op(gate: np.ndarray, n_qubits: int, target: int) -> np.ndarray:
    """Applies a single qubit gate to the target qubit in an n-qubit system."""
    ops = []
    for q in range(n_qubits):
        if q == target:
            ops.append(gate)
        else:
            ops.append(I2)
    res = ops[0]
    for m in ops[1:]:
        res = np.kron(res, m)
    return res

class QuantumSimulator:
    """N-qubit state-vector simulator."""
    def __init__(self, n_qubits: int = 3):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.reset()

    def reset(self):
        """Initializes to |00...0> state."""
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0
        return self

    def apply_gate(self, gate: np.ndarray, target: int):
        op = single_qubit_op(gate, self.n_qubits, target)
        self.state = op @ self.state
        return self

    def apply_cnot(self, control: int, target: int):
        op = cnot_gate(self.n_qubits, control, target)
        self.state = op @ self.state
        return self

    def get_probabilities(self) -> np.ndarray:
        return np.abs(self.state) ** 2

    def get_bloch_coords(self, qubit_idx: int) -> Dict[str, float]:
        """Calculates single-qubit reduced density matrix and Bloch sphere coordinates (x, y, z)."""
        dim = self.dim
        rho = np.outer(self.state, np.conj(self.state))
        
        # Reshape into tensor of shape (2,2,...,2, 2,2,...,2)
        shape = [2] * (2 * self.n_qubits)
        rho_tensor = rho.reshape(shape)
        
        axes_to_trace = [q for q in range(self.n_qubits) if q != qubit_idx]
        reduced = rho_tensor
        for offset, q in enumerate(axes_to_trace):
            adj_q = q - offset
            adj_q_conj = adj_q + (self.n_qubits - offset)
            reduced = np.trace(reduced, axis1=adj_q, axis2=adj_q_conj)
            
        rho_q = reduced.reshape(2, 2)
        
        bx = float(np.real(np.trace(rho_q @ X)))
        by = float(np.real(np.trace(rho_q @ Y)))
        bz = float(np.real(np.trace(rho_q @ Z)))
        
        return {"x": round(bx, 4), "y": round(by, 4), "z": round(bz, 4)}

def generate_bell_states() -> Dict[str, Any]:
    """
    Generates the exact Bell States from Table 5 in Base1.pdf:
    Q0: (|00> + |11>) / sqrt(2)  (Beta_0)
    Q1: (|01> + |10>) / sqrt(2)  (Beta_1)
    Q2: (|00> - |11>) / sqrt(2)  (Beta_2)
    Q3: (|01> - |10>) / sqrt(2)  (Beta_3)
    """
    results = {}
    
    # Q0: Beta 0 = (|00> + |11>)/sqrt(2)
    sim = QuantumSimulator(n_qubits=2)
    sim.apply_gate(H, 0)
    sim.apply_cnot(0, 1)
    probs_0 = sim.get_probabilities()
    results["Q0"] = {
        "name": "Beta 0 (Bell State 1)",
        "formula": "(|00> + |11>) / √2",
        "state_vector": [str(round(x.real, 4) + round(x.imag, 4)*1j) for x in sim.state],
        "amplitudes": [{"basis": f"|{i:02b}>", "prob": round(float(probs_0[i]), 4), "real": round(float(sim.state[i].real), 4), "imag": round(float(sim.state[i].imag), 4)} for i in range(4)]
    }

    # Q1: Beta 1 = (|01> + |10>)/sqrt(2)
    sim.reset()
    sim.apply_gate(X, 1)
    sim.apply_gate(H, 0)
    sim.apply_cnot(0, 1)
    probs_1 = sim.get_probabilities()
    results["Q1"] = {
        "name": "Beta 1 (Bell State 2)",
        "formula": "(|01> + |10>) / √2",
        "state_vector": [str(round(x.real, 4) + round(x.imag, 4)*1j) for x in sim.state],
        "amplitudes": [{"basis": f"|{i:02b}>", "prob": round(float(probs_1[i]), 4), "real": round(float(sim.state[i].real), 4), "imag": round(float(sim.state[i].imag), 4)} for i in range(4)]
    }

    # Q2: Beta 2 = (|00> - |11>)/sqrt(2)
    sim.reset()
    sim.apply_gate(Z, 0)
    sim.apply_gate(H, 0)
    sim.apply_cnot(0, 1)
    probs_2 = sim.get_probabilities()
    results["Q2"] = {
        "name": "Beta 2 (Bell State 3)",
        "formula": "(|00> - |11>) / √2",
        "state_vector": [str(round(x.real, 4) + round(x.imag, 4)*1j) for x in sim.state],
        "amplitudes": [{"basis": f"|{i:02b}>", "prob": round(float(probs_2[i]), 4), "real": round(float(sim.state[i].real), 4), "imag": round(float(sim.state[i].imag), 4)} for i in range(4)]
    }

    # Q3: Beta 3 = (|01> - |10>)/sqrt(2)
    sim.reset()
    sim.apply_gate(X, 1)
    sim.apply_gate(Z, 0)
    sim.apply_gate(H, 0)
    sim.apply_cnot(0, 1)
    probs_3 = sim.get_probabilities()
    results["Q3"] = {
        "name": "Beta 3 (Bell State 4)",
        "formula": "(|01> - |10>) / √2",
        "state_vector": [str(round(x.real, 4) + round(x.imag, 4)*1j) for x in sim.state],
        "amplitudes": [{"basis": f"|{i:02b}>", "prob": round(float(probs_3[i]), 4), "real": round(float(sim.state[i].real), 4), "imag": round(float(sim.state[i].imag), 4)} for i in range(4)]
    }

    return results

def simulate_figure_8_circuit(theta1: float = np.pi/4, phi1: float = np.pi/2, lam1: float = 0.0,
                              theta2: float = np.pi/4, phi2: float = np.pi/5, lam2: float = np.pi/2) -> Dict[str, Any]:
    """
    Simulates the exact 3-qubit quantum model from Figure 8 of Base1.pdf:
    1. q[0]: U(pi/4, pi/2, 0)
    2. q[1]: H
    3. CNOT(control=q[0], target=q[1])
    4. CNOT(control=q[1], target=q[2])
    5. q[2]: U(pi/4, pi/5, pi/2)
    6. Measurement of all qubits onto classical register c3
    """
    sim = QuantumSimulator(n_qubits=3)
    
    # Step 1: U gate on q[0]
    u1 = unitary_gate(theta1, phi1, lam1)
    sim.apply_gate(u1, 0)
    
    # Step 2: H gate on q[1]
    sim.apply_gate(H, 1)
    
    # Step 3: CNOT q0 -> q1
    sim.apply_cnot(0, 1)
    
    # Step 4: CNOT q1 -> q2
    sim.apply_cnot(1, 2)
    
    # Step 5: U gate on q[2]
    u2 = unitary_gate(theta2, phi2, lam2)
    sim.apply_gate(u2, 2)
    
    probs = sim.get_probabilities()
    
    bloch_q0 = sim.get_bloch_coords(0)
    bloch_q1 = sim.get_bloch_coords(1)
    bloch_q2 = sim.get_bloch_coords(2)
    
    basis_states = []
    for i in range(8):
        basis_str = f"|{i:03b}>"
        val = sim.state[i]
        basis_states.append({
            "basis": basis_str,
            "probability": round(float(probs[i]), 5),
            "amplitude_real": round(float(val.real), 5),
            "amplitude_imag": round(float(val.imag), 5),
            "magnitude": round(float(np.abs(val)), 5),
            "phase_deg": round(float(np.angle(val, deg=True)), 2)
        })
        
    return {
        "circuit_name": "Base1.pdf Figure 8 Quantum Model",
        "qubits": 3,
        "classical_bits": 3,
        "parameters": {
            "u1": {"theta": round(theta1, 4), "phi": round(phi1, 4), "lambda": round(lam1, 4)},
            "u2": {"theta": round(theta2, 4), "phi": round(phi2, 4), "lambda": round(lam2, 4)}
        },
        "gates_applied": [
            {"qubit": 0, "gate": f"U({round(theta1,3)}, {round(phi1,3)}, {round(lam1,3)})"},
            {"qubit": 1, "gate": "Hadamard (H)"},
            {"control": 0, "target": 1, "gate": "CNOT(q0 -> q1)"},
            {"control": 1, "target": 2, "gate": "CNOT(q1 -> q2)"},
            {"qubit": 2, "gate": f"U({round(theta2,3)}, {round(phi2,3)}, {round(lam2,3)})"},
            {"qubits": [0, 1, 2], "gate": "Measure to c3"}
        ],
        "basis_states": basis_states,
        "bloch_vectors": {
            "q0": bloch_q0,
            "q1": bloch_q1,
            "q2": bloch_q2
        },
        "entanglement_entropy": round(float(-np.sum([p * np.log2(p) for p in probs if p > 1e-12])), 4)
    }

def quantum_feature_map(x_vec: np.ndarray) -> np.ndarray:
    """
    Quantum feature map for tabular data (QSVC) using 4-qubit parameterized state encoding.
    Maps feature vector into quantum Hilbert space using H, parameter rotations, and CNOT entanglement.
    """
    n_q = 4
    sim = QuantumSimulator(n_qubits=n_q)
    for q in range(n_q):
        sim.apply_gate(H, q)
    for q in range(min(n_q, len(x_vec))):
        angle = float(x_vec[q] * np.pi)
        u = unitary_gate(angle, angle / 2.0, 0.0)
        sim.apply_gate(u, q)
    sim.apply_cnot(0, 1)
    sim.apply_cnot(1, 2)
    sim.apply_cnot(2, 3)
    sim.apply_cnot(3, 0)
    
    return sim.state
