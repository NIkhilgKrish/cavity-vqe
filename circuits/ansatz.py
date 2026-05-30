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

References
----------
Kandala, A. et al. "Hardware-efficient variational quantum eigensolver for
small molecules and quantum magnets." Nature 549, 242–246 (2017).
https://doi.org/10.1038/nature23879
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
    n_params = 2 * n_qubits * (reps + 1)

    @cudaq.kernel
    def ansatz(thetas: List[float]):
        q = cudaq.qvector(n_qubits)
        theta_counter = 0

        # Initial R layer
        for i in range(n_qubits):
            ry(thetas[theta_counter], q[i])
            rz(thetas[theta_counter+1], q[i])
            theta_counter += 2

        # Entangling + R layers
        for r in range(reps):
            # CNOT ring (simplified coupling)
            for i in range(n_qubits):
                cx(q[i], q[(i + 1) % n_qubits])
            
            # R layer
            for i in range(n_qubits):
                ry(thetas[theta_counter + 2 * i], q[i])
                ry(thetas[theta_counter + 2 * i + 1], q[i])
            theta_counter += 2*n_qubits

    return ansatz, n_params

# Setting standard seed for reproducablity
def initial_params(n_params: int, seed: int = 42) -> List[float]:
    """Return a reproducible random initial parameter vector in [-π, π]."""
    import numpy as np
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, size=n_params).tolist()
