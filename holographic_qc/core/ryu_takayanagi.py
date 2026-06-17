from __future__ import annotations

import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


@dataclass
class RTConfig:
    central_charge: float
    newton_constant_3d: float
    ads_radius: float
    uv_cutoff: float = 1e-10


class RyuTakayanagi:
    def __init__(self, config: RTConfig):
        self.c = config.central_charge
        self.G3 = config.newton_constant_3d
        self.L = config.ads_radius
        self.a = config.uv_cutoff

    def geodesic_length_poincare(self, interval_length: float) -> float:
        if interval_length <= self.a:
            raise ValueError("Interval must exceed UV cutoff")
        return 2.0 * self.L * np.log(interval_length / self.a)

    def entanglement_entropy(self, interval_length: float) -> float:
        L_gamma = self.geodesic_length_poincare(interval_length)
        return L_gamma / (4.0 * self.G3)

    def entanglement_entropy_central_charge(self, interval_length: float) -> float:
        return (self.c / 3.0) * np.log(interval_length / self.a)

    def mutual_information(self, l1: float, l2: float, sep: float) -> float:
        S_A = self.entanglement_entropy_central_charge(l1)
        S_B = self.entanglement_entropy_central_charge(l2)
        S_AB = self.entanglement_entropy_two_intervals(l1, l2, sep)
        return S_A + S_B - S_AB

    def entanglement_entropy_two_intervals(
        self, l1: float, l2: float, sep: float
    ) -> float:
        phase1 = (
            self.entanglement_entropy_central_charge(l1) +
            self.entanglement_entropy_central_charge(l2)
        )
        phase2 = (
            self.entanglement_entropy_central_charge(l1 + sep + l2) +
            self.entanglement_entropy_central_charge(sep)
        )
        return min(phase1, phase2)

    def renyi_entropy(self, interval_length: float, n: int) -> float:
        if n == 1:
            return self.entanglement_entropy_central_charge(interval_length)
        prefactor = self.c * (n + 1) / (6.0 * n)
        return prefactor * np.log(interval_length / self.a)

    def holographic_c_function(self, scale: float) -> float:
        return self.c

    def entanglement_entropy_finite_temperature(
        self, interval_length: float, temperature: float
    ) -> float:
        kB = 1.380649e-23
        hbar = 1.054571817e-34
        beta = hbar / (kB * temperature)
        beta_over_pi = beta / np.pi
        arg = beta_over_pi * np.sinh(np.pi * interval_length / beta)
        return (self.c / 3.0) * np.log(arg / self.a)

    def entanglement_entropy_finite_size(
        self, interval_length: float, system_size: float
    ) -> float:
        from scipy.special import gamma as gamma_fn
        arg = (system_size / np.pi) * np.sin(np.pi * interval_length / system_size)
        return (self.c / 3.0) * np.log(arg / self.a)

    def topological_entanglement_entropy(self) -> float:
        return np.log(self.c)

    def ryu_takayanagi_with_quantum_corrections(
        self, interval_length: float, bulk_entropy: float
    ) -> float:
        classical = self.entanglement_entropy(interval_length)
        return classical + bulk_entropy

    def geodesic_arc_length_numerical(
        self, x1: float, z1: float, x2: float, z2: float, n_points: int = 1000
    ) -> float:
        t = np.linspace(0, 1, n_points)
        xc = x1 + (x2 - x1) * t
        zc = z1 + (z2 - z1) * t
        dxdt = (x2 - x1) * np.ones(n_points)
        dzdt = (z2 - z1) * np.ones(n_points)
        integrand = self.L / zc * np.sqrt(dxdt**2 + dzdt**2)
        return float(integrate.trapezoid(integrand, t))

    def minimal_surface_embedding(
        self, boundary_interval: Tuple[float, float], n_points: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_left, x_right = boundary_interval
        ell = x_right - x_left
        x_center = (x_left + x_right) / 2.0
        radius = ell / 2.0
        theta = np.linspace(1e-6, np.pi - 1e-6, n_points)
        x = x_center + radius * np.cos(theta)
        z = radius * np.sin(theta)
        return x, z

    def entanglement_wedge_nesting(
        self, region_A: Tuple[float, float], region_B: Tuple[float, float]
    ) -> bool:
        a1, a2 = region_A
        b1, b2 = region_B
        return a1 >= b1 and a2 <= b2

    def code_distance_holographic(self, d_std: int, system_size: float, lattice_spacing: float) -> float:
        correction = (self.c / (6.0 * np.pi)) * np.log(system_size / lattice_spacing)
        return d_std * (1.0 + correction)

    def error_threshold_holographic(self, p_thresh_std: float) -> float:
        return p_thresh_std * (1.0 + self.c / (12.0 * np.pi))

    def entanglement_spectrum(
        self, interval_length: float, n_eigenvalues: int = 50
    ) -> np.ndarray:
        S = self.entanglement_entropy_central_charge(interval_length)
        xi_n = np.arange(1, n_eigenvalues + 1)
        epsilon_n = 2.0 * np.pi * xi_n / interval_length
        return np.exp(-epsilon_n)

    def modular_hamiltonian_eigenvalues(
        self, interval_length: float, n_levels: int = 20
    ) -> np.ndarray:
        beta_modular = 2.0 * np.pi * interval_length
        n = np.arange(1, n_levels + 1)
        return beta_modular * n / interval_length