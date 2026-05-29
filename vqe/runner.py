"""
VQE runner for CavityVQE.

Uses cudaq.observe() for expectation value evaluation and scipy's COBYLA
for classical optimisation — the same optimizer used in IsingFlow, making
cross-project comparisons straightforward.

Usage
-----
    from vqe.runner import run_vqe
    result = run_vqe()
    print(result["energy"])
"""

from __future__ import annotations
import time
import numpy as np
from scipy.optimize import minimize
import cudaq

from hamiltonian.jaynes_cummings import build as build_hamiltonian, num_qubits, exact_eigenvalues
from circuits.ansatz import make_ansatz, initial_params
import config


def run_vqe(
    omega_c: float     = config.CAVITY_FREQ,
    omega_0: float     = config.ATOM_FREQ,
    g: float           = config.COUPLING,
    n_photon_max: int  = config.N_PHOTON_MAX,
    reps: int          = 2,
    max_iter: int      = config.MAX_ITERATIONS,
    tol: float         = config.TOLERANCE,
    backend: str       = config.BACKEND,
    seed: int          = 42,
) -> dict:
    """Run VQE for the Jaynes-Cummings Hamiltonian.

    Parameters
    ----------
    omega_c, omega_0, g : JC physical parameters
    n_photon_max        : Fock space truncation
    reps                : ansatz repetition layers
    max_iter, tol       : COBYLA termination criteria
    backend             : CUDA-Q simulator backend
    seed                : RNG seed for reproducibility

    Returns
    -------
    dict with keys:
        energy          : float  — final VQE energy
        exact_gs        : float  — exact ground-state energy (numpy diag)
        error           : float  — |VQE - exact|
        params          : list   — optimised parameters
        history         : list   — energy at each iteration
        n_qubits        : int
        n_params        : int
        elapsed_s       : float
    """
    cudaq.set_target(backend)

    n_q     = num_qubits(n_photon_max)
    H       = build_hamiltonian(omega_c, omega_0, g, n_photon_max)
    kernel, n_p = make_ansatz(n_q, reps)
    theta0  = initial_params(n_p, seed)

    history: list[float] = []

    def cost(thetas):
        thetas_list = thetas.tolist()
        result = cudaq.observe(kernel, H, thetas_list)
        e = result.expectation()
        history.append(e)
        return e

    t0 = time.perf_counter()
    opt = minimize(
        cost,
        x0     = np.array(theta0),
        method = "COBYLA",
        options = {"maxiter": max_iter, "rhobeg": 0.5, "catol": tol},
    )
    elapsed = time.perf_counter() - t0

    exact_gs = exact_eigenvalues(omega_c, omega_0, g, n_photon_max)[0]

    return {
        "energy":    opt.fun,
        "exact_gs":  exact_gs,
        "error":     abs(opt.fun - exact_gs),
        "params":    opt.x.tolist(),
        "history":   history,
        "n_qubits":  n_q,
        "n_params":  n_p,
        "elapsed_s": elapsed,
        "converged": opt.success,
    }
