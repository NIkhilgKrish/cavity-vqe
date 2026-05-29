"""
Hardware-efficient ansatz for CavityVQE.

Architecture
------------
For an n-qubit system with `reps` repetitions:

    Layer 0:  Ry(θ) on every qubit           (n parameters)
    ─ repeat reps times ─
      CNOT ring:  CNOT(q_i → q_{i+1 mod n}) for i in 0..n-1
      Ry(θ) on every qubit                   (n parameters per rep)

Total parameters: n * (reps + 1)

This ansatz is expressible on any qubit topology and matches what
NVIDIA's CUDA-Q examples use for hardware-efficient VQE demos.
"""

import cudaq
from cudaq import spin
from typing import List


def make_ansatz(n_qubits: int, reps: int = 2):
    """Return a CUDA-Q kernel for the hardware-efficient ansatz.

    Parameters
    ----------
    n_qubits : total number of qubits
    reps     : number of CNOT + Ry repetition layers

    Returns
    -------
    kernel : cudaq kernel (callable with parameter list)
    n_params : int — number of variational parameters
    """
    n_params = n_qubits * (reps + 1)

    @cudaq.kernel
    def ansatz(thetas: List[float]):
        q = cudaq.qvector(n_qubits)

        # Initial Ry layer
        for i in range(n_qubits):
            ry(thetas[i], q[i])

        # Entangling + Ry layers
        for r in range(reps):
            offset = n_qubits * (r + 1)
            # CNOT ring
            for i in range(n_qubits):
                cx(q[i], q[(i + 1) % n_qubits])
            # Ry layer
            for i in range(n_qubits):
                ry(thetas[offset + i], q[i])

    return ansatz, n_params


def initial_params(n_params: int, seed: int = 42) -> List[float]:
    """Return a reproducible random initial parameter vector in [-π, π]."""
    import numpy as np
    rng = np.random.default_rng(seed)
    return rng.uniform(-3.14159, 3.14159, size=n_params).tolist()
