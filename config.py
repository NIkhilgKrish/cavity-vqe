"""
CavityVQE — Central configuration.

All physical and simulation parameters live here. Override via environment
variables or pass kwargs directly to the JC Hamiltonian / VQE runner.
"""

import os

# ── Physical parameters (all in dimensionless units; set ℏ = 1) ───────────
CAVITY_FREQ   = float(os.getenv("CAVITY_FREQ",   "1.0"))   # ωc
ATOM_FREQ     = float(os.getenv("ATOM_FREQ",     "1.0"))   # ω0  (resonance: ωc == ω0)
COUPLING      = float(os.getenv("COUPLING",      "0.1"))   # g   (weak-coupling regime)

# ── Fock space truncation ──────────────────────────────────────────────────
# Max photon number included in the qubit encoding.
# n_photon_max=1  →  2 qubits total (1 photon qubit + 1 atom qubit)
# n_photon_max=3  →  3 qubits total (2 photon qubits + 1 atom qubit)
N_PHOTON_MAX  = int(os.getenv("N_PHOTON_MAX",    "1"))

# ── VQE optimiser ─────────────────────────────────────────────────────────
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "200"))
TOLERANCE      = float(os.getenv("TOLERANCE",    "1e-6"))
SHOTS          = int(os.getenv("SHOTS",          "0"))     # 0 = exact statevector

# ── CUDA-Q backend ────────────────────────────────────────────────────────
# "qpp-cpu"  → CPU statevector (always available, no GPU needed)
# "nvidia"   → GPU statevector via cuStateVec (requires NVIDIA GPU)
BACKEND = os.getenv("BACKEND", "qpp-cpu")
