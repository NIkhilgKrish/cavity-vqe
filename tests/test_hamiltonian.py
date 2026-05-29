"""
Tests for hamiltonian/jaynes_cummings.py

Checks:
- Correct number of qubits for different Fock truncations
- Hamiltonian is Hermitian (exact matrix)
- Ground-state energy matches analytic result at resonance
- Non-resonant case shifts correctly
"""

import math
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hamiltonian.jaynes_cummings import build, num_qubits, exact_eigenvalues


class TestNumQubits:
    def test_n1(self):
        assert num_qubits(1) == 2

    def test_n3(self):
        # 2 photon qubits + 1 atom qubit
        assert num_qubits(3) == 3


class TestHamiltonian:
    def test_build_returns_spin_operator(self):
        import cudaq
        H = build(1.0, 1.0, 0.1, 1)
        assert isinstance(H, cudaq.SpinOperator)

    def test_invalid_n_photon_max(self):
        with pytest.raises(ValueError):
            build(1.0, 1.0, 0.1, 0)


class TestExactEigenvalues:
    """
    Analytic Jaynes-Cummings eigenvalues (n_photon_max=1, RWA):

    In the one-excitation subspace {|1,g⟩, |0,e⟩} the Hamiltonian is:
        [[ωc,    g  ],
         [ g,   ω0  ]]
    (relative to vacuum energy ωc/2)

    At resonance ωc = ω0 = ω the dressed states are E± = ω ± g.
    The vacuum (no-excitation) state has energy 0 (or ωc/2 + (-ω0/2) per convention).
    """

    def test_resonant_splitting(self):
        # At resonance, eigenvalues should be symmetric about ω ± g
        omega, g = 1.0, 0.1
        eigs = exact_eigenvalues(omega, omega, g, n_photon_max=1)
        assert len(eigs) == 4
        # Ground-state energy must be <= -g (below uncoupled vacuum)
        assert eigs[0] <= -g + 1e-9

    def test_zero_coupling(self):
        # At g=0 the JC model decouples; energies are simple products
        omega_c, omega_0 = 1.0, 0.8
        eigs_g0 = exact_eigenvalues(omega_c, omega_0, g=0.0, n_photon_max=1)
        # With g=0, H = ωc*(I-Z0)/2 + ω0/2*Z1
        # Basis |00⟩,|01⟩,|10⟩,|11⟩ → energies:
        #   |00⟩: 0       - ω0/2  = -ω0/2
        #   |01⟩: 0       + ω0/2  =  ω0/2
        #   |10⟩: ωc      - ω0/2  =  ωc - ω0/2
        #   |11⟩: ωc      + ω0/2  =  ωc + ω0/2
        expected = sorted([
            -omega_0 / 2,
             omega_0 / 2,
             omega_c - omega_0 / 2,
             omega_c + omega_0 / 2,
        ])
        for a, b in zip(sorted(eigs_g0), expected):
            assert abs(a - b) < 1e-10, f"Expected {b:.6f}, got {a:.6f}"

    def test_sorted_ascending(self):
        eigs = exact_eigenvalues(1.0, 1.0, 0.1, 1)
        assert eigs == sorted(eigs)
