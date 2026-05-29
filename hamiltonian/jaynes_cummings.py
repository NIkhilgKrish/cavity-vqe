"""
Jaynes-Cummings Hamiltonian → cudaq.SpinOperator

Physics
-------
The Jaynes-Cummings model (rotating-wave approximation) describes a
two-level atom coupled to a single mode of an electromagnetic cavity:

    H = ωc · a†a  +  ω0/2 · σz  +  g · (a†σ⁻ + a·σ⁺)

where ωc is the cavity frequency, ω0 the atomic transition frequency,
g the light-matter coupling, a†/a photon creation/annihilation operators,
and σ±, σz the atomic Pauli operators.

Qubit encoding (N_photon_max = 1, two-qubit system)
----------------------------------------------------
Fock space is truncated to {|0⟩, |1⟩} for the photon mode.

    qubit 0  →  photon mode   (|0⟩ = vacuum, |1⟩ = one photon)
    qubit 1  →  atom          (|0⟩ = ground, |1⟩ = excited)

Under this encoding:
    a†a        =  (I - Z₀) / 2
    σz         =  Z₁
    a†σ⁻ + aσ⁺ =  (X₀X₁ + Y₀Y₁) / 2

Full Hamiltonian in Pauli basis:
    H = ωc/2·(I - Z₀)  +  ω0/2·Z₁  +  g/2·(X₀X₁ + Y₀Y₁)

The constant ωc/2·I shifts the zero of energy and is kept for correctness.

For N_photon_max > 1 the photon mode requires ceil(log2(N_max+1)) qubits
and the Hamiltonian is built via the Gray-code / binary encoding.  This
module supports both cases but the 2-qubit case is the default and is
what we benchmark in the notebooks.
"""

import math
import cudaq
from cudaq import spin


def build(
    omega_c: float = 1.0,
    omega_0: float = 1.0,
    g: float       = 0.1,
    n_photon_max: int = 1,
) -> cudaq.SpinOperator:
    """Return the JC Hamiltonian as a cudaq.SpinOperator.

    Parameters
    ----------
    omega_c      : cavity (photon) frequency  (ℏ = 1)
    omega_0      : atomic transition frequency (ℏ = 1)
    g            : light-matter coupling strength
    n_photon_max : maximum photon number included in Fock truncation

    Returns
    -------
    cudaq.SpinOperator
        Hamiltonian suitable for use with cudaq.observe().

    Raises
    ------
    ValueError
        If n_photon_max < 1.
    """
    if n_photon_max < 1:
        raise ValueError("n_photon_max must be >= 1")

    if n_photon_max == 1:
        return _build_two_qubit(omega_c, omega_0, g)
    else:
        return _build_multi_qubit(omega_c, omega_0, g, n_photon_max)


def num_qubits(n_photon_max: int = 1) -> int:
    """Total number of qubits required for a given Fock truncation."""
    photon_qubits = math.ceil(math.log2(n_photon_max + 1))
    atom_qubits   = 1
    return photon_qubits + atom_qubits


# ── Private helpers ────────────────────────────────────────────────────────

def _build_two_qubit(omega_c: float, omega_0: float, g: float) -> cudaq.SpinOperator:
    """2-qubit JC Hamiltonian (N_photon_max = 1).

    H = ωc/2·(I - Z₀) + ω0/2·Z₁ + g/2·(X₀X₁ + Y₀Y₁)
    """
    # Energy constant from photon vacuum: ωc/2 · I
    H  = (omega_c / 2.0) * spin.i(0) * spin.i(1)
    # Photon number: -ωc/2 · Z₀
    H += (-omega_c / 2.0) * spin.z(0)
    # Atomic inversion: ω0/2 · Z₁
    H += (omega_0 / 2.0) * spin.z(1)
    # Jaynes-Cummings coupling: g/2 · (X₀X₁ + Y₀Y₁)
    H += (g / 2.0) * spin.x(0) * spin.x(1)
    H += (g / 2.0) * spin.y(0) * spin.y(1)
    return H


def _build_multi_qubit(
    omega_c: float, omega_0: float, g: float, n_photon_max: int
) -> cudaq.SpinOperator:
    """Multi-qubit JC Hamiltonian via binary Fock encoding.

    Photon mode occupies qubits [0 .. n_ph-1] (little-endian binary),
    atom occupies qubit n_ph.

    For n_photon_max = 3 → 2 photon qubits + 1 atom qubit = 3 qubits.

    The photon number operator in binary encoding is:
        n̂ = Σ_k  2^k · (I - Z_k) / 2   for k in [0, n_ph)

    The interaction terms aσ⁺ + a†σ⁻ are built by summing ladder
    operators in the truncated Fock basis and decomposing them into Paulis.
    """
    n_ph = math.ceil(math.log2(n_photon_max + 1))
    atom = n_ph  # qubit index for the atom

    # ── Photon number operator: ωc · n̂ ────────────────────────────────────
    H = cudaq.SpinOperator()
    for k in range(n_ph):
        coeff = omega_c * (2 ** k) / 2.0
        H += coeff * spin.i(k)   # constant part (shifts energy zero)
        H -= coeff * spin.z(k)

    # ── Atomic inversion: ω0/2 · σz ───────────────────────────────────────
    H += (omega_0 / 2.0) * spin.z(atom)

    # ── Interaction: g · (a†σ⁻ + h.c.) ───────────────────────────────────
    # Decompose photon ladder operators in binary Fock basis as Pauli strings.
    # For each adjacent pair |n⟩ ↔ |n+1⟩ (n < n_photon_max) the contribution
    # to a is sqrt(n+1) · |n⟩⟨n+1|.  In the binary register this maps to a
    # tensor product of X and Y operators whose exact form depends on the
    # bit-flip pattern between the binary representations of n and n+1.
    # We build these explicitly for small truncations.
    for n in range(n_photon_max):
        matrix_element = math.sqrt(n + 1)
        ket   = n       # |n⟩
        bra   = n + 1   # ⟨n+1|

        # Bits that differ between ket and bra
        diff_bits = ket ^ bra
        flip_qubits = [k for k in range(n_ph) if (diff_bits >> k) & 1]

        # |ket⟩⟨bra| expressed in Paulis for each flipped qubit:
        #   |0⟩⟨1| = (X + iY)/2  →  contributes X and Y Pauli strings
        # For this XY decomposition we build the full term as a product.
        # Simple case: if exactly one bit flips (always true for sequential
        # Fock states in standard binary), the interaction is local.
        if len(flip_qubits) == 1:
            k = flip_qubits[0]
            # a · σ⁺ = |ket⟩⟨bra| ⊗ |g⟩⟨e|
            # Pauli decomposition of |0⟩⟨1| ⊗ |0⟩⟨1| = (X⊗X + X⊗Y·i - Y⊗X·i + Y⊗Y)/4
            # but we add h.c. so imaginary parts cancel:
            # (a†σ⁻ + aσ⁺) = g/2·(X_k X_atom + Y_k Y_atom) (same form as 2-qubit case)
            H += (g * matrix_element / 2.0) * spin.x(k) * spin.x(atom)
            H += (g * matrix_element / 2.0) * spin.y(k) * spin.y(atom)

    return H


def exact_eigenvalues(
    omega_c: float = 1.0,
    omega_0: float = 1.0,
    g: float       = 0.1,
    n_photon_max: int = 1,
) -> list[float]:
    """Return exact eigenvalues via NumPy diagonalisation (reference only).

    Useful for benchmarking VQE accuracy.
    """
    import numpy as np

    H_op = build(omega_c, omega_0, g, n_photon_max)
    n    = num_qubits(n_photon_max)
    dim  = 2 ** n
    H_matrix = np.zeros((dim, dim), dtype=complex)

    for term in H_op:
        coeff   = term.get_coefficient()
        pauli_w = term.get_pauli_word()   # e.g. "XIYZ"
        mat     = _pauli_word_to_matrix(pauli_w, n)
        H_matrix += coeff * mat

    eigvals = np.linalg.eigvalsh(H_matrix.real)
    return sorted(eigvals.tolist())


def _pauli_word_to_matrix(word: str, n: int):
    """Convert a Pauli word string (LSB-first) to a dense matrix."""
    import numpy as np

    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    pauli = {"I": I, "X": X, "Y": Y, "Z": Z}

    # word is qubit-0 first; build tensor product from qubit n-1 down to 0
    mats = [pauli[c] for c in word.upper()]
    # pad to n qubits if shorter
    while len(mats) < n:
        mats.append(I)
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(m, result)
    return result
