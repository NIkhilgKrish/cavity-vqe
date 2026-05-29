"""
Tests for vqe/runner.py and circuits/ansatz.py

Checks:
- Ansatz produces the right parameter count
- VQE converges to within 1e-3 of exact ground state (qpp-cpu backend)
- Result dict has all expected keys
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from circuits.ansatz import make_ansatz, initial_params
from vqe.runner import run_vqe


class TestAnsatz:
    def test_param_count_2q_reps2(self):
        _, n_p = make_ansatz(n_qubits=2, reps=2)
        assert n_p == 2 * (2 + 1)   # n_qubits * (reps + 1) = 6

    def test_param_count_3q_reps1(self):
        _, n_p = make_ansatz(n_qubits=3, reps=1)
        assert n_p == 3 * (1 + 1)   # 6

    def test_initial_params_length(self):
        params = initial_params(6, seed=0)
        assert len(params) == 6

    def test_initial_params_reproducible(self):
        p1 = initial_params(10, seed=7)
        p2 = initial_params(10, seed=7)
        assert p1 == p2


class TestVQERunner:
    """Integration tests — run actual VQE on qpp-cpu (no GPU needed)."""

    def test_result_keys(self):
        result = run_vqe(omega_c=1.0, omega_0=1.0, g=0.1, backend="qpp-cpu", max_iter=50)
        expected_keys = {"energy", "exact_gs", "error", "params", "history",
                         "n_qubits", "n_params", "elapsed_s", "converged"}
        assert expected_keys.issubset(result.keys())

    def test_convergence_resonant(self):
        """At resonance with weak coupling, VQE should hit < 1e-3 error."""
        result = run_vqe(omega_c=1.0, omega_0=1.0, g=0.1, backend="qpp-cpu",
                         max_iter=200, reps=2)
        assert result["error"] < 1e-3, (
            f"VQE error {result['error']:.4e} exceeds threshold 1e-3"
        )

    def test_history_non_empty(self):
        result = run_vqe(omega_c=1.0, omega_0=1.0, g=0.05, backend="qpp-cpu", max_iter=30)
        assert len(result["history"]) > 0

    def test_energy_below_excited(self):
        """VQE ground state must be below the first excited state."""
        from hamiltonian.jaynes_cummings import exact_eigenvalues
        result = run_vqe(omega_c=1.0, omega_0=1.0, g=0.1, backend="qpp-cpu", max_iter=150)
        eigs = exact_eigenvalues(1.0, 1.0, 0.1, 1)
        assert result["energy"] < eigs[1] + 1e-4
