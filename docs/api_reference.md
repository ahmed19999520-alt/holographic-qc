# API Reference

## `holographic_qc.core`

### `VirasoroAlgebra`

```python
VirasoroAlgebra(config: VirasoroConfig)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `commutator_scalar(m, n)` | `(float, float)` | Linear and central coefficients of $[L_m, L_n]$ |
| `ope_tilde_coeff(z, w)` | `complex` | Leading OPE coefficient $c/2/(z-w)^4$ |
| `two_point_function(z1, z2, h)` | `complex` | $\langle\mathcal{O}(z_1)\mathcal{O}(z_2)\rangle = (z_1-z_2)^{-2h}$ |
| `character(h, q, n_levels)` | `complex` | Virasoro character $\chi_h(q)$ |
| `kac_table(p, q)` | `np.ndarray` | Kac table of conformal weights |
| `gram_matrix(h, level)` | `np.ndarray` | Gram matrix at given descendant level |
| `kac_determinant(h, level)` | `float` | Kac determinant (vanishes at degenerate representations) |
| `lyapunov_from_central_charge(T)` | `float` | $\lambda_L = (2\pi k_B T/\hbar)(1 - 6/c^2)$ |

### `AdsCft3`

```python
AdsCft3(central_charge, ads_radius, newton_constant_3d=None, fermi_velocity=5e5)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `bulk_to_boundary_propagator(z, x, x_prime, delta)` | `float` | $K(z,x;x') = (z/(z^2+(x-x')^2))^\Delta$ |
| `two_point_function(x, x_prime, delta)` | `float` | $C_\Delta/|x-x'|^{2\Delta}$ |
| `optical_conductivity_dc()` | `float` | $\sigma_0 = (e^2/h)(c/2)$ |
| `wiedemann_franz_ratio()` | `float` | $\mathcal{L}_0(1 - 3/c)$ |
| `lyapunov_exponent(T)` | `float` | Holographic Lyapunov exponent |
| `scrambling_time(T, n)` | `float` | $t_s = (\hbar/2\pi k_B T)\ln N$ |
| `entanglement_entropy(ell, uv)` | `float` | $(c/3)\ln(\ell/a)$ |
| `holographic_coherence_enhancement(L, xi)` | `float` | $(L/\xi)^{c/6}$ |

### `ChristoffelSymbols`

```python
ChristoffelSymbols(metric: np.ndarray)
ChristoffelSymbols.from_ads3_poincare(ads_radius, z)
ChristoffelSymbols.from_sphere(radius, theta)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `compute(dg)` | `np.ndarray` | $\Gamma^\sigma_{\mu\nu}$ from metric derivatives |
| `riemann_tensor(dg, d2g)` | `np.ndarray` | Full Riemann tensor |
| `ricci_tensor(dg, d2g)` | `np.ndarray` | Ricci contraction |
| `geodesic_equation(x0, v0, params, metric_fn)` | `np.ndarray` | Geodesic trajectory |
| `ads3_geodesic_length(x1,z1,x2,z2,L)` | `float` | $L_\gamma = L\cosh^{-1}(1+\sigma)$ |
| `ads3_geodesic_length_boundary(ell,a,L)` | `float` | $2L\ln(\ell/a)$ |
| `symbolic_christoffel(g_sym, coords)` | `sp.Array` | Symbolic Christoffel symbols via SymPy |

### `RyuTakayanagi`

```python
RyuTakayanagi(config: RTConfig)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `entanglement_entropy(ell)` | `float` | $L_\gamma/(4G_3)$ |
| `entanglement_entropy_central_charge(ell)` | `float` | $(c/3)\ln(\ell/a)$ |
| `renyi_entropy(ell, n)` | `float` | $(c(n+1)/6n)\ln(\ell/a)$ |
| `mutual_information(l1, l2, sep)` | `float` | $I(A:B) = S_A + S_B - S_{AB}$ |
| `entanglement_entropy_finite_temperature(ell, T)` | `float` | Thermal EE |
| `code_distance_holographic(d, L, a)` | `float` | $d(1 + (c/6\pi)\ln(L/a))$ |

---

## `holographic_qc.algorithms`

### `ShorAlgorithm`

```python
ShorAlgorithm(N: int, n_precision: int = None)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `factor(max_attempts)` | `Optional[(int,int)]` | Full quantum-classical Shor |
| `factor_classical_simulation(max_attempts)` | `Optional[(int,int)]` | Classical period finding |
| `factors_from_period(a, r)` | `Optional[(int,int)]` | Extract factors from period |
| `circuit_resource_estimate()` | `dict` | Gate counts and qubit requirements |
| `success_probability_estimate(a)` | `float` | Theoretical success probability |
| `verify_factorization(p, q)` | `bool` | Verify $p \cdot q = N$ |

### `GroverAlgorithm`

```python
GroverAlgorithm(n_qubits: int, target_states: List[int])
```

| Method | Returns | Description |
|--------|---------|-------------|
| `optimal_iterations()` | `int` | $k^* = \lfloor\pi/(4\theta)\rfloor$ |
| `success_probability(k)` | `float` | $\sin^2((2k+1)\theta)$ |
| `run(n_iterations)` | `(state, int)` | Run Grover circuit |
| `run_with_measurement(n_shots)` | `dict` | Run and sample measurements |
| `amplitude_amplification(state, oracle, k)` | `np.ndarray` | General amplitude amplification |
| `circuit_resource_estimate()` | `dict` | Gate counts and speedup factor |

### `QuantumFourierTransform`

```python
QuantumFourierTransform(n_qubits: int)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `matrix()` | `np.ndarray` | Full $2^n \times 2^n$ DFT matrix |
| `apply(state)` | `np.ndarray` | Apply QFT via matrix multiply |
| `apply_fft(state)` | `np.ndarray` | Apply via NumPy FFT |
| `estimate_phase(state)` | `float` | Phase estimation via inverse QFT |
| `verify_unitarity(tol)` | `bool` | Check $F F^\dagger = I$ |

---

## `holographic_qc.protection`

### `HolographicDecoherence`

```python
HolographicDecoherence(ads_system: AdsCft3, material, config=None)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `standard_phonon_rate_2d(T)` | `float` | $\gamma \propto \alpha T^2$ |
| `holographic_decoherence_rate(T, L, xi)` | `float` | Screened rate \eqref{eq:gamma_holo} |
| `coherence_time_ratio(L, xi)` | `float` | $(L/\xi)^{c/6}$ |
| `coherence_time_holographic(T, L)` | `float` | $T_2^{\rm std} \times (L/\xi)^{c/6}$ |
| `temperature_dependence_enhancement(T_arr, L)` | `np.ndarray` | Temperature scan |
| `combined_topological_holographic_T2(T2, L, xi)` | `float` | $T_2 e^{L/\xi}(L/\xi)^{c/6}$ |

### `MajoranaQubit`

```python
MajoranaQubit(wire_length, coherence_length, fermi_velocity=5e5, ads_system=None)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `parity_operator()` | `np.ndarray` | $\mathcal{P} = i\sigma^z$ |
| `braiding_unitary(i, j)` | `np.ndarray` | $U_{ij} = e^{i\pi\gamma_i\gamma_j/4}$ |
| `holographic_berry_phase()` | `float` | $\pi/4 \cdot (1 + (\xi/L)\ln(L/\xi))$ |
| `gate_fidelity(epsilon0)` | `float` | $1 - \epsilon_0(L/\xi)^{-c/6}$ |
| `total_coherence_time(tau0)` | `float` | Combined protection formula |
| `energy_splitting()` | `float` | Majorana hybridization energy |

---

## `holographic_qc.materials`

| Class | $c$ | $v_F$ [m/s] | $\xi$ [nm] | Platform |
|-------|-----|------------|-----------|---------|
| `Bi2Se3` | 1 | $5\times10^5$ | 1.10 | STM |
| `HgTeCdTe` | 2 | $3\times10^5$ | 19.8 | Transport |
| `TrappedIonChain` | 0.5 | — | — | Ions |

---

## `holographic_qc.ml`

### TensorFlow: `HolographicDecoherenceNet`

Architecture: Feature encoder → VirasoroEquivariantLayer → BulkPropagatorNet → {decoherence_head, enhancement_head}

### PyTorch: `HolographicDecoherenceNet`

Architecture: Batch-norm residual blocks → VirasoroEquivariantLayer → Dual output heads

Both models output:
```python
{
    "decoherence_rate": Tensor,   # units: s^{-1}
    "enhancement_factor": Tensor  # dimensionless
}
```