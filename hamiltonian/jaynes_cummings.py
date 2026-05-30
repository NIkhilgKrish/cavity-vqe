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

Sign convention: σz has the EXCITED atomic state as its +1 eigenvalue
(standard physics convention, excited state higher in energy).  Since the
computational |1⟩ is the excited atom and Z|1⟩ = -1, the atomic operator is
σz = -Z on the atom qubit.

Under this encoding:
    a†a        =  (I - Z₀) / 2
    σz         =  -Z₁                       (excited |1⟩ → +1)
    a†σ⁻ + aσ⁺ =  (X₀X₁ + Y₀Y₁) / 2

Full Hamiltonian in Pauli basis:
    H = ωc/2·(I - Z₀)  -  ω0/2·Z₁  +  g/2·(X₀X₁ + Y₀Y₁)

The constant ωc/2·I shifts the zero of energy and is kept for correctness.

For N_photon_max > 1 the photon mode requires ceil(log2(N_max+1)) qubits.
The photon Fock number is stored in qubits [0 .. n_ph-1] (little-endian
binary, qubit 0 = least-significant bit) and the atom occupies the highest
qubit, index n_ph.  The Hamiltonian is assembled directly from truncated
ladder operators a, a†, n̂ = a†a (decomposed into Pauli strings on the
photon qubits only) combined with the exact single-qubit atom operators
σ±, σz — rather than decomposing the full 2^N×2^N matrix.  Fock indices
beyond n_photon_max (present only because 2^n_ph may exceed n_photon_max+1)
are pushed out of the physical spectrum by a large diagonal penalty.
"""

import math
import numpy as np
import itertools
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

    H = ωc/2·(I - Z₀) - ω0/2·Z₁ + g/2·(X₀X₁ + Y₀Y₁)

    qubit 0 = photon, qubit 1 = atom (highest qubit). σz = -Z₁ so the
    excited atom |1⟩ has energy +ω0/2.
    """
    # Energy constant from photon vacuum: ωc/2 · I
    H  = (omega_c / 2.0) * spin.i(0)
    # Photon number: -ωc/2 · Z₀
    H += (-omega_c / 2.0) * spin.z(0)
    # Atomic inversion ω0/2·σz with σz = -Z₁ (excited |1⟩ → +1): -ω0/2 · Z₁
    H += (-omega_0 / 2.0) * spin.z(1)
    # Jaynes-Cummings coupling: g/2 · (X₀X₁ + Y₀Y₁)
    H += (g / 2.0) * spin.x(0) * spin.x(1)
    H += (g / 2.0) * spin.y(0) * spin.y(1)
    return H


def _build_multi_qubit(
    omega_c: float,
    omega_0: float,
    g: float,
    n_photon_max: int,
    penalty: float = 1.0e6,
) -> cudaq.SpinOperator:
    """Multi-qubit JC Hamiltonian via binary Fock encoding.

    Photon Fock number occupies qubits [0 .. n_ph-1] (little-endian binary,
    qubit 0 = LSB); the atom occupies the highest qubit, index n_ph.
    For n_photon_max = 3 → 2 photon qubits + 1 atom qubit = 3 qubits.

    Built directly from truncated ladder operators rather than by decomposing
    the full system matrix:

        H = ωc·n̂ + (ω0/2)·σz + g·(a†σ⁻ + aσ⁺)                      (+ penalty)

    n̂ = a†a, a, a† are dense (2^n_ph × 2^n_ph) matrices truncated at
    n_photon_max and decomposed into Pauli strings on the photon qubits only.
    The atom operators σz = -Z (excited |1⟩ → +1), σ⁺ = (X - iY)/2,
    σ⁻ = (X + iY)/2 are exact single-qubit Paulis on qubit n_ph.

    Because the binary register holds 2^n_ph states but only n_photon_max + 1
    are physical, any leftover Fock indices receive a large diagonal `penalty`
    (× identity on the atom) so they cannot appear as low-lying eigenstates.
    """
    n_ph    = math.ceil(math.log2(n_photon_max + 1))
    atom    = n_ph                 # atom is the highest qubit
    photon  = list(range(n_ph))    # photon qubits 0 .. n_ph-1 (qubit 0 = LSB)
    D       = 2 ** n_ph            # photon register dimension

    # Truncated bosonic ladder operators on the photon register.
    a = np.zeros((D, D), dtype=complex)
    for f in range(1, n_photon_max + 1):       # a|f⟩ = √f |f-1⟩, f ≤ n_photon_max
        a[f - 1, f] = math.sqrt(f)
    adag  = a.conj().T
    n_hat = adag @ a                            # diag(0,1,...,n_photon_max,0,...)

    # Penalty projector onto unphysical (truncated-away) Fock indices.
    pen = np.zeros((D, D), dtype=complex)
    for f in range(n_photon_max + 1, D):
        pen[f, f] = penalty

    # Photon-space operators → SpinOperators on the photon qubits.
    n_op    = _matrix_to_spin(n_hat, photon)
    a_op    = _matrix_to_spin(a,     photon)
    adag_op = _matrix_to_spin(adag,  photon)
    pen_op  = _matrix_to_spin(pen,   photon)

    # Exact single-qubit atom operators on qubit `atom`.
    sigma_z     = -1.0 * spin.z(atom)                       # excited |1⟩ → +1
    sigma_plus  = 0.5 * (spin.x(atom) - 1j * spin.y(atom))  # |e⟩⟨g|
    sigma_minus = 0.5 * (spin.x(atom) + 1j * spin.y(atom))  # |g⟩⟨e|

    H  = omega_c * n_op
    H += (omega_0 / 2.0) * sigma_z
    H += g * (adag_op * sigma_minus + a_op * sigma_plus)
    if pen_op is not None:
        H += pen_op
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

    # H_matrix is Hermitian; eigvalsh handles the complex case directly.
    eigvals = np.linalg.eigvalsh(H_matrix)
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

def _matrix_to_spin(mat, qubits, tol: float = 1e-12):
    """Decompose a 2^k × 2^k matrix into a cudaq.SpinOperator on `qubits`.

    `qubits[j]` carries Pauli factor j, with qubits[0] the least-significant
    bit — matching `_pauli_word_to_matrix`.  Works for non-Hermitian `mat`
    (e.g. ladder operators), producing complex Pauli coefficients via the
    Hilbert-Schmidt inner product  c_P = tr(P·mat) / 2^k.

    Returns None if every coefficient is below `tol` (so callers can skip an
    all-zero operator instead of relying on an empty SpinOperator).
    """
    spin_fn = {"I": spin.i, "X": spin.x, "Y": spin.y, "Z": spin.z}
    k   = len(qubits)
    dim = 2 ** k
    assert mat.shape == (dim, dim), "matrix size does not match qubit count"

    result = None
    for label in itertools.product("IXYZ", repeat=k):
        word  = "".join(label)
        coeff = np.trace(_pauli_word_to_matrix(word, k) @ mat) / dim
        if abs(coeff) < tol:
            continue
        term = spin_fn[label[0]](qubits[0])
        for j in range(1, k):
            term = term * spin_fn[label[j]](qubits[j])
        contribution = coeff * term
        result = contribution if result is None else result + contribution
    return result
