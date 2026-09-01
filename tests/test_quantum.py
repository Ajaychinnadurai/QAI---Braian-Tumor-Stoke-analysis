"""
Unit Tests for Quantum Circuit Simulator, Bell States & Fig. 8 Circuit
"""

import unittest
import numpy as np
from backend.quantum_engine import (
    QuantumSimulator, generate_bell_states, simulate_figure_8_circuit,
    unitary_gate, H, X, Z
)

class TestQuantumEngine(unittest.TestCase):
    def test_bell_states_table_5(self):
        bell = generate_bell_states()
        self.assertIn("Q0", bell)
        self.assertIn("Q1", bell)
        self.assertIn("Q2", bell)
        self.assertIn("Q3", bell)

        # For Beta 0: (|00> + |11>)/sqrt(2), |00> and |11> probs should be 0.5, others 0.0
        q0_probs = [a["prob"] for a in bell["Q0"]["amplitudes"]]
        self.assertAlmostEqual(q0_probs[0], 0.5, places=2)
        self.assertAlmostEqual(q0_probs[1], 0.0, places=2)
        self.assertAlmostEqual(q0_probs[2], 0.0, places=2)
        self.assertAlmostEqual(q0_probs[3], 0.5, places=2)

    def test_figure_8_simulation(self):
        res = simulate_figure_8_circuit()
        self.assertEqual(res["qubits"], 3)
        self.assertEqual(len(res["basis_states"]), 8)

        # Probability sum must equal 1.0
        total_prob = sum(s["probability"] for s in res["basis_states"])
        self.assertAlmostEqual(total_prob, 1.0, places=4)

        # Bloch coordinates must be within unit sphere: x^2 + y^2 + z^2 <= 1.0001
        for q_name, coords in res["bloch_vectors"].items():
            r2 = coords["x"]**2 + coords["y"]**2 + coords["z"]**2
            self.assertLessEqual(r2, 1.001)

if __name__ == "__main__":
    unittest.main()
