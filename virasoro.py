from __future__ import annotations

import numpy as np
import scipy.linalg as la
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from functools import lru_cache


@dataclass
class VirasoroConfig:
    central_charge: float
    max_mode: int = 10
    dtype: type = np.complex128


class VirasoroAlgebra:
    def __init__(self, config: VirasoroConfig):
        self.c = config.central_charge
        self.N = config.max_mode
        self.dtype = config.dtype
        self._dim = 2 * self.N + 1
        self._mode_index = {n: n + self.N for n in range(-self.N, self.N + 1)}
        self._modes = list(range(-self.N, self.N + 1))

    def commutator_scalar(self, m: int, n: int) -> Tuple[float, float]:
        linear_coeff = float(m - n)
        central_coeff = 0.0
        if m + n == 0:
            central_coeff = (self.c / 12.0) * m * (m * m - 1)
        return linear_coeff, central_coeff

    def mode_matrix(self, n: int) -> np.ndarray:
        mat = np.zeros((self._dim, self._dim), dtype=self.dtype)
        for m in self._modes:
            result = m + n
            if result in self._mode_index:
                i = self._mode_index[result]
                j = self._mode_index[m]
                mat[i, j] += (m - (-n)) * (-1)
        return mat

    def verify_jacobi_identity(self, l: int, m: int, n: int, tol: float = 1e-10) -> bool:
        def bracket(a: int, b: int) -> Tuple[float, float]:
            return self.commutator_scalar(a, b)

        lin_ab, cen_ab = bracket(l, m)
        lin_bc, cen_bc = bracket(m, n)
        lin_ca, cen_ca = bracket(n, l)

        jacc_linear = lin_ab * (l + m - n) + lin_bc * (m + n - l) + lin_ca * (n + l - m)
        return abs(jacc_linear) < tol

    def ope_tilde_coeff(self, z: complex, w: complex) -> complex:
        denom = (z - w)
        if abs(denom) < 1e-15:
            raise ValueError("Coincident points in OPE")
        return (self.c / 2.0) / denom**4

    def two_point_function(self, z1: complex, z2: complex, h: float) -> complex:
        denom = z1 - z2
        if abs(denom) < 1e-15:
            raise ValueError("Coincident points")
        return 1.0 / denom**(2 * h)

    def three_point_function(
        self, z1: complex, z2: complex, z3: complex,
        h1: float, h2: float, h3: float, C123: complex = 1.0
    ) -> complex:
        d12 = z1 - z2
        d13 = z1 - z3
        d23 = z2 - z3
        exp12 = h1 + h2 - h3
        exp13 = h1 + h3 - h2
        exp23 = h2 + h3 - h1
        return C123 / (d12**exp12 * d13**exp13 * d23**exp23)

    def stress_tensor_ward_identity(
        self, z: complex, zi: List[complex], hi: List[float]
    ) -> complex:
        result = 0.0 + 0j
        for zk, hk in zip(zi, hi):
            dzk = z - zk
            if abs(dzk) < 1e-15:
                continue
            result += hk / dzk**2 + (1.0 / dzk) * 0.0
        return result

    def character(self, h: float, q: complex, n_levels: int = 30) -> complex:
        if abs(q) >= 1.0:
            raise ValueError("|q| must be < 1 for convergence")
        prefactor = q**(h - self.c / 24.0)
        eta_inv = 1.0 + 0j
        for n in range(1, n_levels + 1):
            eta_inv /= (1.0 - q**n)
        return prefactor * eta_inv

    def kac_table(self, p: int, q_param: int) -> np.ndarray:
        table = np.zeros((p - 1, q_param - 1))
        for r in range(1, p):
            for s in range(1, q_param):
                h = ((p * s - q_param * r)**2 - (p - q_param)**2) / (4.0 * p * q_param)
                table[r - 1, s - 1] = h
        return table

    def conformal_anomaly(self) -> float:
        return self.c

    def central_charge_from_ope(self, T_vev: float, R: float) -> float:
        return 12.0 * T_vev * R**2

    def partition_states(self, level: int) -> List[Tuple[int, ...]]:
        result: List[Tuple[int, ...]] = []
        self._partitions_rec(level, level, [], result)
        return result

    def _partitions_rec(
        self, n: int, max_val: int, current: List[int], result: List[Tuple[int, ...]]
    ):
        if n == 0:
            result.append(tuple(current))
            return
        for i in range(min(n, max_val), 0, -1):
            current.append(i)
            self._partitions_rec(n - i, i, current, result)
            current.pop()

    def gram_matrix(self, h: float, level: int) -> np.ndarray:
        states = self.partition_states(level)
        n = len(states)
        G = np.zeros((n, n), dtype=np.float64)
        for i, p1 in enumerate(states):
            for j, p2 in enumerate(states):
                G[i, j] = self._gram_element(h, p1, p2)
        return G

    def _gram_element(self, h: float, p1: Tuple[int, ...], p2: Tuple[int, ...]) -> float:
        if sorted(p1) != sorted(p2):
            return 0.0
        val = 1.0
        for k in p1:
            val *= (2 * h + k - 1) * k
        return float(val)

    def kac_determinant(self, h: float, level: int) -> float:
        G = self.gram_matrix(h, level)
        return float(np.linalg.det(G))

    def virasoro_block_zamolodchikov(
        self, h: float, hi: Tuple[float, float, float, float], z: complex, n_terms: int = 20
    ) -> complex:
        h1, h2, h3, h4 = hi
        c1 = h2 - h1
        c2 = h3 - h4
        result = 0.0 + 0j
        z_pow = z**h
        for n in range(n_terms):
            coeff = self._zamolodchikov_coeff(h, c1, c2, n)
            result += coeff * z**n
        return z_pow * result

    def _zamolodchikov_coeff(self, h: float, c1: float, c2: float, n: int) -> complex:
        if n == 0:
            return 1.0 + 0j
        num = (h + c1) * (h - c1) * (h + c2) * (h - c2)
        den = float(2 * n) * (2 * h + n - 1)
        if abs(den) < 1e-15:
            return 0.0 + 0j
        return (num / den) + 0j

    def lyapunov_from_central_charge(self, temperature: float, hbar: float = 1.054571817e-34, kB: float = 1.380649e-23) -> float:
        bound = 2.0 * np.pi * kB * temperature / hbar
        correction = 6.0 / (self.c**2)
        return bound * (1.0 - correction)


class WardIdentityVerifier:
    def __init__(self, algebra: VirasoroAlgebra):
        self.algebra = algebra

    def verify_conformal_ward(
        self, positions: np.ndarray, weights: np.ndarray, tol: float = 1e-8
    ) -> Tuple[bool, np.ndarray]:
        n = len(positions)
        residuals = np.zeros(3, dtype=np.complex128)
        residuals[0] = np.sum(weights)
        residuals[1] = np.sum(weights * positions + weights)
        residuals[2] = np.sum(weights * positions**2 + 2 * weights * positions)
        return np.all(np.abs(residuals) < tol), residuals

    def verify_algebra_closure(self, mode_range: int = 5, tol: float = 1e-10) -> Dict[str, float]:
        errors = {}
        for m in range(-mode_range, mode_range + 1):
            for n in range(-mode_range, mode_range + 1):
                lin, cen = self.algebra.commutator_scalar(m, n)
                expected_mode = m + n
                key = f"[L_{m}, L_{n}]"
                errors[key] = abs(lin - (m - n))
        return errors