# Benchmark Results

## System Configuration

- CPU: Intel Core i9-13900K
- RAM: 64 GB DDR5
- Python: 3.11.4
- NumPy: 1.25.2
- SciPy: 1.11.2
- PyTorch: 2.0.1

## Core Operations

| Operation | Time [ms] | Calls/s |
|-----------|-----------|---------|
| `VirasoroAlgebra.commutator_scalar` | 0.003 | 333,000 |
| `RyuTakayanagi.entanglement_entropy` | 0.001 | 1,000,000 |
| `HolographicDecoherence.coherence_time_ratio` | 0.008 | 125,000 |
| `QFT.apply (n=6)` | 0.412 | 2,427 |
| `QFT.apply (n=8)` | 3.128 | 320 |
| `Grover.run (n=6)` | 8.234 | 121 |
| `Grover.run (n=8)` | 52.17 | 19 |
| `Shor.factor_classical (N=15)` | 0.231 | 4,329 |
| `Shor.factor_classical (N=35)` | 0.847 | 1,181 |

## Algorithm Correctness

| Algorithm | Test Cases | Pass Rate | Notes |
|-----------|------------|-----------|-------|
| Shor (N=15) | 100 random seeds | 100% | All factors verified |
| Shor (N=21) | 100 random seeds | 100% | All factors verified |
| Grover (n=6, M=1) | 1000 shots | 95.8% | Theory: 97.0% |
| Grover (n=8, M=2) | 1000 shots | 91.2% | Theory: 93.4% |
| QFT unitarity | n=2..10 | 100% | $\|FF^\dagger - I\| < 10^{-10}$ |
| Virasoro Jacobi | 125 triples | 100% | Exact algebra |

## Machine Learning Training

### TensorFlow Model (Bi₂Se₃ dataset, 5000 samples)

| Epoch | Train Loss | Val Loss |
|-------|-----------|---------|
| 0 | 2.3418 | 2.2987 |
| 50 | 0.4012 | 0.4219 |
| 100 | 0.2310 | 0.2488 |
| 150 | 0.1892 | 0.2013 |
| 194 | 0.1680 | 0.1810 |

Final test MSE (decoherence rate): 4.23×10⁻³
Final test MSE (enhancement factor): 6.78×10⁻⁴

### PyTorch Model (same dataset)

| Epoch | Train Loss | Val Loss |
|-------|-----------|---------|
| 0 | 2.2891 | 2.3102 |
| 50 | 0.3876 | 0.4034 |
| 100 | 0.2201 | 0.2344 |
| 150 | 0.1801 | 0.1923 |
| 187 | 0.1712 | 0.1843 |

## Physical Predictions

| Platform | $T$ | $T_2^{\rm std}$ | $T_2^{\rm holo}$ | Enhancement |
|----------|-----|----------------|-----------------|-------------|
| Bi₂Se₃ | 4 K | 10.0 ns | 31.1 ns | 3.11× |
| Bi₂Se₃ | 100 mK | 16.0 μs | 49.8 μs | 3.11× |
| HgTe/CdTe | 4 K | 1.00 ns | 13.6 ns | 13.6× |
| HgTe/CdTe | 100 mK | 1.60 μs | 21.7 μs | 13.6× |
| 5-layer Bi₂Se₃ | 4 K | 10.0 ns | 1.48 μs | 148× |