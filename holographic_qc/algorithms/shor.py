from __future__ import annotations

import math
import random
import numpy as np
from typing import Optional, Tuple
from holographic_qc.algorithms.qft import QuantumFourierTransform


class ModularExponentiator:
    def __init__(self, base: int, modulus: int):
        self.a = base
        self.N = modulus

    def compute(self, x: int) -> int:
        return pow(self.a, x, self.N)

    def circuit_unitary(self, n_control: int, n_target: int) -> np.ndarray:
        dim = 2**(n_control + n_target)
        U = np.zeros((dim, dim), dtype=np.complex128)
        N_ctrl = 2**n_control
        N_tgt = 2**n_target
        for ctrl in range(N_ctrl):
            for tgt in range(N_tgt):
                row_in = ctrl * N_tgt + tgt
                out_tgt = (tgt * self.compute(ctrl)) % N_tgt
                row_out = ctrl * N_tgt + out_tgt
                U[row_out, row_in] = 1.0
        return U


class PeriodFinder:
    def __init__(self, a: int, N: int, n_precision: int = None):
        self.a = a
        self.N = N
        self.n_precision = n_precision or math.ceil(2 * math.log2(N))
        self.qft = QuantumFourierTransform(self.n_precision)

    def initial_state(self) -> np.ndarray:
        n_ctrl = self.n_precision
        dim = 2**n_ctrl
        state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
        return state

    def apply_modular_exp(self, ctrl_state: np.ndarray) -> np.ndarray:
        dim = len(ctrl_state)
        phases = np.array([
            np.exp(2j * np.pi * pow(self.a, x, self.N) / self.N)
            for x in range(dim)
        ])
        return ctrl_state * phases

    def measure_frequency(self, state: np.ndarray) -> int:
        probs = np.abs(self.qft.apply(state))**2
        probs /= probs.sum()
        return int(np.argmax(probs))

    def continued_fraction_period(self, measured: int) -> Optional[int]:
        dim = 2**self.n_precision
        p_over_q = measured / dim
        cf = self._continued_fraction(p_over_q, 20)
        convergents = self._convergents(cf)
        for _, q in convergents:
            if q > 0 and q < self.N and pow(self.a, q, self.N) == 1:
                return int(q)
        return None

    def _continued_fraction(self, x: float, max_terms: int) -> list:
        terms = []
        for _ in range(max_terms):
            a_i = int(x)
            terms.append(a_i)
            frac = x - a_i
            if abs(frac) < 1e-10:
                break
            x = 1.0 / frac
        return terms

    def _convergents(self, cf: list) -> list:
        convergents = []
        h_prev, h_curr = 1, cf[0]
        k_prev, k_curr = 0, 1
        convergents.append((h_curr, k_curr))
        for i in range(1, len(cf)):
            h_next = cf[i] * h_curr + h_prev
            k_next = cf[i] * k_curr + k_prev
            convergents.append((h_next, k_next))
            h_prev, h_curr = h_curr, h_next
            k_prev, k_curr = k_curr, k_next
        return convergents

    def run(self, n_trials: int = 10) -> Optional[int]:
        state = self.initial_state()
        state = self.apply_modular_exp(state)
        for _ in range(n_trials):
            m = self.measure_frequency(state)
            r = self.continued_fraction_period(m)
            if r is not None and r > 0:
                return r
        return None


class ShorAlgorithm:
    def __init__(self, N: int, n_precision: int = None):
        self.N = N
        self.n_precision = n_precision

    def is_prime_classical(self, n: int) -> bool:
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    def is_prime_power(self, n: int) -> Tuple[bool, Optional[int], Optional[int]]:
        for b in range(2, int(math.log2(n)) + 1):
            a = round(n**(1.0 / b))
            for candidate in [a - 1, a, a + 1]:
                if candidate > 1 and candidate**b == n:
                    return True, candidate, b
        return False, None, None

    def classical_gcd_factor(self) -> Optional[Tuple[int, int]]:
        a = random.randint(2, self.N - 1)
        g = math.gcd(a, self.N)
        if 1 < g < self.N:
            return g, self.N // g
        return None

    def quantum_period_finding(self, a: int) -> Optional[int]:
        finder = PeriodFinder(a, self.N, self.n_precision)
        return finder.run()

    def factors_from_period(self, a: int, r: int) -> Optional[Tuple[int, int]]:
        if r % 2 != 0:
            return None
        x = pow(a, r // 2, self.N)
        if x == self.N - 1:
            return None
        f1 = math.gcd(x + 1, self.N)
        f2 = math.gcd(x - 1, self.N)
        if 1 < f1 < self.N:
            return f1, self.N // f1
        if 1 < f2 < self.N:
            return f2, self.N // f2
        return None

    def factor(self, max_attempts: int = 20) -> Optional[Tuple[int, int]]:
        if self.N % 2 == 0:
            return 2, self.N // 2

        is_pp, base, exp = self.is_prime_power(self.N)
        if is_pp:
            return base, self.N // base

        for _ in range(max_attempts):
            classical = self.classical_gcd_factor()
            if classical is not None:
                return classical

            a = random.randint(2, self.N - 1)
            g = math.gcd(a, self.N)
            if 1 < g < self.N:
                return g, self.N // g

            r = self.quantum_period_finding(a)
            if r is None:
                continue

            result = self.factors_from_period(a, r)
            if result is not None:
                return result

        return None

    def factor_classical_simulation(self, max_attempts: int = 100) -> Optional[Tuple[int, int]]:
        for _ in range(max_attempts):
            a = random.randint(2, self.N - 1)
            g = math.gcd(a, self.N)
            if 1 < g < self.N:
                return g, self.N // g
            r = 1
            x = a % self.N
            while x != 1 and r < self.N:
                x = (x * a) % self.N
                r += 1
            if r < self.N and r % 2 == 0:
                result = self.factors_from_period(a, r)
                if result is not None:
                    return result
        return None

    def circuit_resource_estimate(self) -> dict:
        n = math.ceil(math.log2(self.N))
        n_precision = self.n_precision or 2 * n
        return {
            "n_logical_qubits": 2 * n + n_precision,
            "n_qft_gates": n_precision * (n_precision + 1) // 2,
            "n_modular_exp_gates": n_precision * n**2,
            "circuit_depth": n_precision * n**2 + n_precision * (n_precision + 1) // 2,
            "classical_post_processing_ops": n_precision * int(math.log2(n_precision)),
        }

    def success_probability_estimate(self, a: int) -> float:
        r = 1
        x = a % self.N
        while x != 1 and r < self.N:
            x = (x * a) % self.N
            r += 1
        if r >= self.N:
            return 0.0
        euler_phi = sum(1 for k in range(1, self.N) if math.gcd(k, self.N) == 1)
        return 0.5 * (1.0 - 1.0 / r) if euler_phi > 0 else 0.0

    def verify_factorization(self, p: int, q: int) -> bool:
        return p * q == self.N and p > 1 and q > 1