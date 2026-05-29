"""
CavityVQE — CLI entrypoint.

Usage
-----
    python run.py                          # default parameters from config.py
    python run.py --g 0.2 --reps 3        # override coupling and ansatz depth
    python run.py --sweep                  # run Rabi splitting sweep + save plots
"""

import argparse
import os

from vqe.runner import run_vqe
from analysis.plots import (
    plot_convergence,
    plot_rabi_splitting,
    plot_energy_spectrum,
    print_summary,
    save_or_show,
)
from hamiltonian.jaynes_cummings import exact_eigenvalues
import config


def main():
    parser = argparse.ArgumentParser(description="CavityVQE: JC model ground state via CUDA-Q")
    parser.add_argument("--omega_c",      type=float, default=config.CAVITY_FREQ)
    parser.add_argument("--omega_0",      type=float, default=config.ATOM_FREQ)
    parser.add_argument("--g",            type=float, default=config.COUPLING)
    parser.add_argument("--n_photon_max", type=int,   default=config.N_PHOTON_MAX)
    parser.add_argument("--reps",         type=int,   default=2)
    parser.add_argument("--backend",      type=str,   default=config.BACKEND)
    parser.add_argument("--sweep",        action="store_true",
                        help="Also plot vacuum Rabi splitting sweep")
    parser.add_argument("--save",         type=str,   default=None,
                        help="Directory to save plots (omit to display interactively)")
    args = parser.parse_args()

    print(f"\nCavityVQE  |  backend: {args.backend}")
    print(f"JC params  |  ωc={args.omega_c}  ω₀={args.omega_0}  g={args.g}\n")

    result = run_vqe(
        omega_c      = args.omega_c,
        omega_0      = args.omega_0,
        g            = args.g,
        n_photon_max = args.n_photon_max,
        reps         = args.reps,
        backend      = args.backend,
    )
    print_summary(result)

    exact = exact_eigenvalues(args.omega_c, args.omega_0, args.g, args.n_photon_max)

    save_dir = args.save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fig1 = plot_convergence(result["history"], result["exact_gs"])
    save_or_show(fig1, os.path.join(save_dir, "convergence.png") if save_dir else None)

    fig2 = plot_energy_spectrum(result["energy"], exact)
    save_or_show(fig2, os.path.join(save_dir, "spectrum.png") if save_dir else None)

    if args.sweep:
        import numpy as np
        g_vals = np.linspace(0, 0.5, 80).tolist()
        fig3 = plot_rabi_splitting(args.omega_c, args.omega_0, g_vals, args.n_photon_max)
        save_or_show(fig3, os.path.join(save_dir, "rabi_splitting.png") if save_dir else None)


if __name__ == "__main__":
    main()
