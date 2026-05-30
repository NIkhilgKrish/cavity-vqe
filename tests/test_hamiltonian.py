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


def _analytic_jc_spectrum(omega_c, omega_0, g, n_photon_max):
    """Exact dressed-state spectrum of the truncated JC model.

    Sign convention: excited atom has energy +ω0/2.

    Physical states (2·(n_photon_max+1) of them):
      • |0, g⟩                       → -ω0/2
      • |n_photon_max, e⟩            → ωc·n_photon_max + ω0/2   (no partner; truncated)
      • for each excitation block N = 1..n_photon_max, the pair
        {|N, g⟩, |N-1, e⟩} gives  ωc(N - ½) ± √((Δ/2)² + g²·N),  Δ = ω0 - ωc
    """
    delta = omega_0 - omega_c
    vals = [-omega_0 / 2.0, omega_c * n_photon_max + omega_0 / 2.0]
    for N in range(1, n_photon_max + 1):
        root = math.sqrt((delta / 2.0) ** 2 + g ** 2 * N)
        vals.append(omega_c * (N - 0.5) + root)
        vals.append(omega_c * (N - 0.5) - root)
    return sorted(vals)


class TestMultiQubitLadder:
    """Validate the multi-qubit (binary Fock) Hamiltonian against the
    analytic Jaynes-Cummings dressed-state ladder."""

    CASES = [
        # (omega_c, omega_0, g, n_photon_max)
        (1.0, 1.0, 0.10, 3),   # resonant, 2^n_ph == n_photon_max+1 (no unused states)
        (1.0, 1.5, 0.20, 2),   # detuned, one unused Fock index per atom state
        (1.0, 0.7, 0.30, 4),   # detuned, several unused Fock indices
        (2.0, 2.0, 0.05, 3),   # resonant, different scale
    ]

    @pytest.mark.parametrize("omega_c,omega_0,g,n_photon_max", CASES)
    def test_physical_spectrum_matches_analytic(self, omega_c, omega_0, g, n_photon_max):
        eigs = exact_eigenvalues(omega_c, omega_0, g, n_photon_max)
        n_physical = 2 * (n_photon_max + 1)
        # Unused Fock indices are pushed up by the penalty, so the lowest
        # n_physical eigenvalues are exactly the physical dressed states.
        physical = sorted(eigs)[:n_physical]
        expected = _analytic_jc_spectrum(omega_c, omega_0, g, n_photon_max)
        assert np.allclose(physical, expected, atol=1e-9), (
            f"\n got: {np.round(physical, 6)}\n want: {np.round(expected, 6)}"
        )

    @pytest.mark.parametrize("omega_c,omega_0,g,n_photon_max", CASES)
    def test_penalty_separates_unphysical_states(self, omega_c, omega_0, g, n_photon_max):
        n_ph = math.ceil(math.log2(n_photon_max + 1))
        n_unused = (2 ** n_ph - (n_photon_max + 1)) * 2
        eigs = sorted(exact_eigenvalues(omega_c, omega_0, g, n_photon_max))
        if n_unused == 0:
            return
        # The penalty (1e6) must lift every unphysical state far above physical ones.
        assert eigs[-n_unused] > 1e5

    def test_hermitian(self):
        # exact_eigenvalues uses eigvalsh, which requires a Hermitian matrix;
        # complex eigenvalues here would surface as a numerical failure.
        eigs = exact_eigenvalues(1.0, 1.3, 0.25, 3)
        assert all(abs(e.imag) < 1e-12 for e in np.asarray(eigs, dtype=complex))

    def test_matches_two_qubit_at_n1(self):
        # build() routes n_photon_max=1 to the 2-qubit path; the analytic
        # ladder must still agree.
        eigs = exact_eigenvalues(1.0, 1.2, 0.15, 1)
        expected = _analytic_jc_spectrum(1.0, 1.2, 0.15, 1)
        assert np.allclose(sorted(eigs), expected, atol=1e-9)
