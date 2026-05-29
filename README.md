# CavityVQE

**Variational Quantum Eigensolver for the Jaynes-Cummings model, implemented in NVIDIA CUDA-Q.**

Finds the ground-state energy of a two-level atom coupled to a single cavity photon mode — a foundational quantum optics system — using a hardware-efficient ansatz and COBYLA optimisation on CUDA-Q's statevector simulator.

---

## Key Results

| Metric | Value |
|--------|-------|
| Model | Jaynes-Cummings (RWA, 1-photon Fock truncation) |
| Qubits | 2 |
| Ansatz | Hardware-efficient, 2 reps, 6 parameters |
| Framework | NVIDIA CUDA-Q (`qpp-cpu` / `nvidia` backends) |
| VQE energy | converges to exact GS within ~1e-4 |
| Vacuum Rabi splitting | reproduced at resonance (gap ≈ 2g) |

---

## Physics

The Jaynes-Cummings Hamiltonian (rotating-wave approximation, ℏ = 1):

```
H = ωc·a†a + ω0/2·σz + g·(a†σ⁻ + a·σ⁺)
```

Qubit encoding (2-qubit, Fock truncation N=1):

```
H = ωc/2·(I − Z₀) + ω0/2·Z₁ + g/2·(X₀X₁ + Y₀Y₁)
```

At resonance (ωc = ω0), the dressed states split by ±g — the **vacuum Rabi splitting**.

---

## Project Structure

```
CavityVQE/
├── hamiltonian/
│   └── jaynes_cummings.py   # JC → cudaq.SpinOperator + exact diagonalisation
├── circuits/
│   └── ansatz.py            # Hardware-efficient CUDA-Q kernel
├── vqe/
│   └── runner.py            # VQE loop: cudaq.observe + COBYLA
├── analysis/
│   └── plots.py             # Convergence, Rabi splitting, spectrum plots
├── tests/                   # pytest suite (Hamiltonian + VQE integration)
├── notebooks/
│   └── walkthrough.ipynb    # End-to-end demo with 4 charts
├── config.py                # All parameters, env-var overridable
└── run.py                   # CLI entrypoint
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Run VQE with default parameters (qpp-cpu, no GPU needed)
python run.py

# Override coupling strength and ansatz depth
python run.py --g 0.2 --reps 3

# Run with GPU backend (requires NVIDIA GPU + cudaq[nvidia])
python run.py --backend nvidia

# Run full sweep with Rabi splitting plot
python run.py --sweep --save plots/

# Run tests
pytest tests/ -v
```

---

## Notebooks

Open `notebooks/walkthrough.ipynb` for a step-by-step demo covering:
1. Hamiltonian construction and exact eigenvalues
2. Ansatz circuit diagram
3. VQE convergence
4. Vacuum Rabi splitting sweep
5. Off-resonance detuning scan

---

## Configuration

All parameters are set in `config.py` and overridable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CAVITY_FREQ` | 1.0 | ωc — cavity frequency |
| `ATOM_FREQ` | 1.0 | ω0 — atomic transition frequency |
| `COUPLING` | 0.1 | g — light-matter coupling |
| `N_PHOTON_MAX` | 1 | Fock space truncation |
| `BACKEND` | `qpp-cpu` | CUDA-Q simulator backend |
| `MAX_ITERATIONS` | 200 | COBYLA max iterations |

---

## Stack

- [NVIDIA CUDA-Q](https://nvidia.github.io/cuda-quantum/) — quantum kernel execution and statevector simulation
- NumPy / SciPy — classical optimisation (COBYLA) and exact diagonalisation
- Matplotlib — results visualisation

---

## Related

- [IsingFlow](../IsingFlow) — QAOA-based combinatorial optimisation (MaxCut/QUBO) using Qiskit + ReAct agent
