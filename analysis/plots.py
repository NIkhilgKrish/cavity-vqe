"""
Analysis and plotting for CavityVQE.

Functions
---------
plot_convergence       — VQE energy vs iteration
plot_rabi_splitting    — ground-state energy vs coupling g (vacuum Rabi)
plot_energy_spectrum   — exact vs VQE energy bar chart
save_or_show           — helper to save/display figures
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Style ──────────────────────────────────────────────────────────────────
NVIDIA_GREEN = "#76b900"
DARK_BG      = "#1a1a2e"
TEXT_COLOR   = "#e0e0e0"

def _apply_style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor(DARK_BG)
    ax.figure.patch.set_facecolor(DARK_BG)
    ax.set_title(title, color=TEXT_COLOR, fontsize=13, pad=10)
    ax.set_xlabel(xlabel, color=TEXT_COLOR)
    ax.set_ylabel(ylabel, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")


# ── Public API ─────────────────────────────────────────────────────────────

def plot_convergence(
    history: list[float],
    exact_gs: float,
    title: str = "VQE Energy Convergence — Jaynes-Cummings Model",
) -> plt.Figure:
    """Plot VQE energy vs iteration alongside exact ground-state energy."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    iters = range(1, len(history) + 1)
    ax.plot(iters, history, color=NVIDIA_GREEN, linewidth=1.5, label="VQE energy")
    ax.axhline(exact_gs, color="#ff6b6b", linestyle="--", linewidth=1.2,
               label=f"Exact GS  ({exact_gs:.4f})")
    ax.fill_between(iters, history, exact_gs, alpha=0.12, color=NVIDIA_GREEN)
    ax.legend(facecolor="#2a2a4a", labelcolor=TEXT_COLOR, framealpha=0.8)
    _apply_style(ax, title, "Iteration", "Energy (ℏ = 1)")
    fig.tight_layout()
    return fig


def plot_rabi_splitting(
    omega_c: float = 1.0,
    omega_0: float = 1.0,
    g_values: list[float] | None = None,
    n_photon_max: int = 1,
) -> plt.Figure:
    """Plot exact energy eigenvalues vs coupling g (vacuum Rabi splitting).

    At resonance (ωc = ω0) the two dressed states split by ±g — the
    classic signature of the Jaynes-Cummings model.
    """
    from hamiltonian.jaynes_cummings import exact_eigenvalues

    if g_values is None:
        g_values = np.linspace(0, 0.5, 60).tolist()

    eigvals = [exact_eigenvalues(omega_c, omega_0, g, n_photon_max) for g in g_values]
    eigvals = np.array(eigvals)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [NVIDIA_GREEN, "#4ecdc4", "#ffd166", "#ef476f"]
    for i in range(eigvals.shape[1]):
        ax.plot(g_values, eigvals[:, i], color=colors[i % len(colors)],
                linewidth=1.8, label=f"E_{i}")

    ax.legend(facecolor="#2a2a4a", labelcolor=TEXT_COLOR, framealpha=0.8)
    _apply_style(ax,
        title   = f"Vacuum Rabi Splitting  (ωc = {omega_c}, ω₀ = {omega_0})",
        xlabel  = "Coupling g",
        ylabel  = "Energy (ℏ = 1)",
    )
    fig.tight_layout()
    return fig


def plot_energy_spectrum(
    vqe_energy: float,
    exact_eigenvalues: list[float],
    title: str = "Energy Spectrum — VQE vs Exact Diagonalisation",
) -> plt.Figure:
    """Bar chart comparing VQE ground state to all exact eigenvalues."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n = len(exact_eigenvalues)
    x = np.arange(n)
    bars = ax.bar(x, exact_eigenvalues, color="#4ecdc4", alpha=0.75,
                  label="Exact eigenvalues", zorder=2)
    ax.axhline(vqe_energy, color=NVIDIA_GREEN, linestyle="--", linewidth=2,
               label=f"VQE ground state ({vqe_energy:.4f})", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"E_{i}" for i in range(n)], color=TEXT_COLOR)
    ax.legend(facecolor="#2a2a4a", labelcolor=TEXT_COLOR, framealpha=0.8)
    _apply_style(ax, title, "Eigenstate", "Energy (ℏ = 1)")
    fig.tight_layout()
    return fig


def print_summary(result: dict):
    """Print a concise benchmark summary to stdout."""
    print("\n" + "=" * 52)
    print("  CavityVQE — Jaynes-Cummings Ground State")
    print("=" * 52)
    print(f"  Qubits        : {result['n_qubits']}")
    print(f"  Parameters    : {result['n_params']}")
    print(f"  Iterations    : {len(result['history'])}")
    print(f"  Elapsed       : {result['elapsed_s']:.2f} s")
    print(f"  VQE energy    : {result['energy']:.6f}")
    print(f"  Exact GS      : {result['exact_gs']:.6f}")
    print(f"  |Error|       : {result['error']:.2e}")
    converged_str = "✓" if result.get("converged") else "✗"
    print(f"  Converged     : {converged_str}")
    print("=" * 52 + "\n")


def save_or_show(fig: plt.Figure, path: str | None = None):
    """Save figure to path or display interactively."""
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved: {path}")
    else:
        plt.show()
    plt.close(fig)
