from __future__ import annotations

import numpy as np
import sympy as sp
from typing import Callable, Dict, List, Optional, Tuple


class ChristoffelSymbols:
    def __init__(self, metric: np.ndarray, coordinates: Optional[List[str]] = None):
        self.dim = metric.shape[0]
        if metric.shape != (self.dim, self.dim):
            raise ValueError("Metric must be square")
        self.g = metric.astype(np.float64)
        self.g_inv = np.linalg.inv(self.g)
        self.coords = coordinates or [f"x{i}" for i in range(self.dim)]
        self._gamma = None

    @classmethod
    def from_ads3_poincare(cls, ads_radius: float, z: float) -> ChristoffelSymbols:
        factor = (ads_radius / z)**2
        g = np.diag([factor, factor, -factor])
        return cls(g, coordinates=["x", "z", "t"])

    @classmethod
    def from_sphere(cls, radius: float, theta: float) -> ChristoffelSymbols:
        g = np.diag([radius**2, radius**2 * np.sin(theta)**2])
        return cls(g, coordinates=["theta", "phi"])

    @classmethod
    def from_schwarzschild(cls, mass: float, r: float, c: float = 3e8, G: float = 6.674e-11) -> ChristoffelSymbols:
        rs = 2.0 * G * mass / c**2
        f = 1.0 - rs / r
        g = np.diag([f, 1.0 / f, r**2, r**2])
        g[0, 0] = -f
        return cls(g, coordinates=["t", "r", "theta", "phi"])

    def compute(self, dg: np.ndarray) -> np.ndarray:
        gamma = np.zeros((self.dim, self.dim, self.dim))
        for sigma in range(self.dim):
            for mu in range(self.dim):
                for nu in range(self.dim):
                    val = 0.0
                    for lam in range(self.dim):
                        val += 0.5 * self.g_inv[sigma, lam] * (
                            dg[nu, lam, mu] + dg[mu, lam, nu] - dg[lam, mu, nu]
                        )
                    gamma[sigma, mu, nu] = val
        self._gamma = gamma
        return gamma

    def from_metric_function(
        self, metric_fn: Callable[[np.ndarray], np.ndarray], point: np.ndarray, eps: float = 1e-6
    ) -> np.ndarray:
        g0 = metric_fn(point)
        dg = np.zeros((self.dim, self.dim, self.dim))
        for mu in range(self.dim):
            ep = np.zeros(self.dim)
            ep[mu] = eps
            g_plus = metric_fn(point + ep)
            g_minus = metric_fn(point - ep)
            dg[mu] = (g_plus - g_minus) / (2.0 * eps)
        self.g = g0
        self.g_inv = np.linalg.inv(g0)
        return self.compute(dg)

    def riemann_tensor(self, dg: np.ndarray, d2g: np.ndarray) -> np.ndarray:
        gamma = self.compute(dg)
        R = np.zeros((self.dim, self.dim, self.dim, self.dim))
        for rho in range(self.dim):
            for sigma in range(self.dim):
                for mu in range(self.dim):
                    for nu in range(self.dim):
                        term1 = 0.0
                        term2 = 0.0
                        term3 = 0.0
                        term4 = 0.0
                        for lam in range(self.dim):
                            term3 += gamma[lam, nu, sigma] * gamma[rho, mu, lam]
                            term4 += gamma[lam, mu, sigma] * gamma[rho, nu, lam]
                        R[rho, sigma, mu, nu] = term1 - term2 + term3 - term4
        return R

    def ricci_tensor(self, dg: np.ndarray, d2g: np.ndarray) -> np.ndarray:
        R_full = self.riemann_tensor(dg, d2g)
        Ric = np.zeros((self.dim, self.dim))
        for mu in range(self.dim):
            for nu in range(self.dim):
                for lam in range(self.dim):
                    Ric[mu, nu] += R_full[lam, mu, lam, nu]
        return Ric

    def ricci_scalar(self, dg: np.ndarray, d2g: np.ndarray) -> float:
        Ric = self.ricci_tensor(dg, d2g)
        R = 0.0
        for mu in range(self.dim):
            for nu in range(self.dim):
                R += self.g_inv[mu, nu] * Ric[mu, nu]
        return float(R)

    def geodesic_equation(
        self, x0: np.ndarray, v0: np.ndarray,
        affine_param: np.ndarray,
        metric_fn: Callable[[np.ndarray], np.ndarray], eps: float = 1e-6
    ) -> np.ndarray:
        from scipy.integrate import solve_ivp

        def rhs(lam, state):
            x = state[:self.dim]
            v = state[self.dim:]
            gamma = self.from_metric_function(metric_fn, x, eps)
            dv = np.zeros(self.dim)
            for sigma in range(self.dim):
                for mu in range(self.dim):
                    for nu in range(self.dim):
                        dv[sigma] -= gamma[sigma, mu, nu] * v[mu] * v[nu]
            return np.concatenate([v, dv])

        state0 = np.concatenate([x0, v0])
        sol = solve_ivp(
            rhs, [affine_param[0], affine_param[-1]],
            state0, t_eval=affine_param, rtol=1e-10, atol=1e-12
        )
        return sol.y[:self.dim].T

    def ads3_geodesic_length(
        self, x1: float, z1: float, x2: float, z2: float, ads_radius: float
    ) -> float:
        chordal = ((x1 - x2)**2 + (z1 - z2)**2) / (2.0 * z1 * z2)
        return ads_radius * np.arccosh(1.0 + chordal)

    def ads3_geodesic_length_boundary(self, ell: float, uv_cutoff: float, ads_radius: float) -> float:
        return 2.0 * ads_radius * np.log(ell / uv_cutoff)

    def parallel_transport(
        self, vector: np.ndarray, curve: np.ndarray,
        metric_fn: Callable[[np.ndarray], np.ndarray], eps: float = 1e-6
    ) -> np.ndarray:
        n_points = len(curve)
        transported = np.zeros((n_points, self.dim))
        transported[0] = vector.copy()
        for i in range(n_points - 1):
            dx = curve[i + 1] - curve[i]
            gamma = self.from_metric_function(metric_fn, curve[i], eps)
            dv = np.zeros(self.dim)
            for sigma in range(self.dim):
                for mu in range(self.dim):
                    for nu in range(self.dim):
                        dv[sigma] -= gamma[sigma, mu, nu] * transported[i, mu] * dx[nu]
            transported[i + 1] = transported[i] + dv
        return transported

    def covariant_derivative_vector(
        self, V: np.ndarray, dV: np.ndarray, dg: np.ndarray
    ) -> np.ndarray:
        gamma = self.compute(dg)
        nabla_V = dV.copy()
        for mu in range(self.dim):
            for nu in range(self.dim):
                for sigma in range(self.dim):
                    nabla_V[mu, nu] += gamma[mu, nu, sigma] * V[sigma]
        return nabla_V

    def symbolic_christoffel(self, metric_sym: sp.Matrix, coord_syms: List[sp.Symbol]) -> sp.Array:
        n = len(coord_syms)
        g = metric_sym
        g_inv = g.inv()
        gamma = sp.MutableDenseNDimArray(sp.zeros(n**3), (n, n, n))
        for sigma in range(n):
            for mu in range(n):
                for nu in range(n):
                    val = sp.Integer(0)
                    for lam in range(n):
                        term = sp.Rational(1, 2) * g_inv[sigma, lam] * (
                            sp.diff(g[lam, nu], coord_syms[mu]) +
                            sp.diff(g[lam, mu], coord_syms[nu]) -
                            sp.diff(g[mu, nu], coord_syms[lam])
                        )
                        val += term
                    gamma[sigma, mu, nu] = sp.simplify(val)
        return gamma