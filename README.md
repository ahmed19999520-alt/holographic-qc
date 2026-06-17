# HolographicQC

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CI](https://github.com/ahmed19999520-alt/holographic-qc/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmed19999520-alt/holographic-qc/actions)

A production-grade Python and C# library implementing holographic quantum
computing primitives derived from the AdS₃/CFT₂ correspondence, including the
Virasoro algebra, bulk-to-boundary propagators, holographic decoherence
suppression, Majorana qubit encoding, Shor's algorithm, and Grover's algorithm.
The library is designed to accompany the paper:

> **Holographic Engineering for Topological Quantum Computing and
> Suppression-Protection of Qubit Decoherence**
> *Ahmed Ali, University of Bonn / Max Planck Institute for Physics*

---

## Physical Background

The AdS₃/CFT₂ correspondence maps a three-dimensional gravitational theory in
anti-de Sitter space to a two-dimensional conformal field theory on its boundary.
The helical edge modes of a quantum spin Hall insulator realize this boundary CFT
with central charge $c = 1$ per edge channel. Environmental noise couples to
boundary operators of conformal dimension $\Delta_n$, sources the bulk dilaton
$\phi(z, x, t)$, and is screened by the bulk-to-boundary propagator:

$$K(z, x; x') = \left(\frac{z}{z^2 + (x - x')^2}\right)^{\!\Delta}$$

yielding a coherence-time enhancement:

$$\frac{T_2^{\mathrm{holo}}}{T_2^{\mathrm{std}}} = \left(\frac{L}{\xi}\right)^{c/6}$$

---

## Installation

```bash
git clone https://github.com/ahmed19999520-alt/holographic-qc.git
cd holographic-qc
pip install -e ".[all]"
```

### Optional backends

```bash
pip install tensorflow>=2.12
pip install torch>=2.0
pip install qiskit>=0.44
```

---

## Quickstart

```python
from holographic_qc.core.virasoro import VirasoroAlgebra, VirasoroConfig
from holographic_qc.core.ads_cft import AdsCft3
from holographic_qc.protection.decoherence import HolographicDecoherence
from holographic_qc.materials.bi2se3 import Bi2Se3

mat = Bi2Se3()
ads = AdsCft3(central_charge=mat.central_charge, ads_radius=mat.xi)
dec = HolographicDecoherence(ads_system=ads, material=mat)

ratio = dec.coherence_time_ratio(system_size=1e-6)
print(f"T2_holo / T2_std = {ratio:.4f}")
```
Expected output:
T2_holo / T2_std = 3.1124

---

## Algorithms

| Algorithm | Module | Reference |
|-----------|--------|-----------|
| Shor's factoring | `algorithms.shor` | Shor (1994) |
| Grover's search | `algorithms.grover` | Grover (1996) |
| Quantum Fourier Transform | `algorithms.qft` | Coppersmith (1994) |
| Variational Quantum Eigensolver | `algorithms.vqe` | Peruzzo et al. (2014) |
| Holographic Error Correction | `protection.holographic_error_correction` | Pastawski et al. (2015) |

---

## Training

```bash
python scripts/train_decoherence_tf.py \
    --epochs 200 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --data data/bi2se3_arpes.csv \
    --output models/decoherence_tf/

python scripts/train_decoherence_torch.py \
    --epochs 200 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --data data/bi2se3_arpes.csv \
    --output models/decoherence_torch/
```

---

## Citation

```bibtex
@article{ali2026holographic,
  title   = {Holographic Engineering for Topological Quantum Computing
             and Suppression-Protection of Qubit Decoherence},
  author  = {Ali, Ahmed},
  journal = {Journal of High Energy Physics},
  year    = {2026},
  note    = {Preprint}
}
```