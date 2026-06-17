from __future__ import annotations

import numpy as np
import scipy.integrate as integrate
import scipy.special as special
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


@dataclass
class AdsCft3:
    central_charge: float
    ads_radius: float
    newton_constant_3d: float = None
    fermi_velocity: float = 5e5

    def __post_init__(self):
        if self.newton_constant_3d is None:
            self.newton_constant_3d = 3.0 * self.ads_radius / (2.0 * self.central_charge)

    def brown_henneaux_central_charge(self) -> float:
        return 3.0 * self.ads_radius / (2.0 * self.newton_constant_3d)

    def poincare_metric(self, z: float, x: float, t: float) -> np.ndarray:
        factor = (self.ads_radius / z)**2
        g = np.diag([-factor, factor, factor])
        return g

    def metric_determinant(self, z: float) -> float:
        return -(self.ads_radius / z)**6

    def christoffel_poincare(self, z: float) -> Dict[Tuple, float]:
        symbols = {}
        L = self.ads_radius
        symbols[(0, 0, 2)] = -1.0 / z
        symbols[(1, 1, 2)] = -1.0 / z
        symbols[(2, 2, 2)] = -1.0 / z
        symbols[(2, 0, 0)] = 1.0 / z
        symbols[(2, 1, 1)] = 1.0 / z
        return symbols

    def bulk_to_boundary_propagator(
        self, z: float, x: float, x_prime: float, delta: float
    ) -> float:
        r_sq = z**2 + (x - x_prime)**2
        return (z / r_sq)**delta

    def bulk_to_boundary_propagator_fourier(
        self, z: float, k: float, nu: float
    ) -> complex:
        kz = abs(k) * z
        return (kz)**(0.5) * special.kv(nu, kz) * (abs(k))**(nu - 0.5)

    def bulk_to_bulk_propagator(
        self, z1: float, x1: float, z2: float, x2: float, delta: float
    ) -> float:
        u = ((z1 - z2)**2 + (x1 - x2)**2) / (2.0 * z1 * z2)
        xi = 1.0 / (1.0 + u)
        result = xi**delta * special.hyp2f1(delta, delta - 0.5, 2.0 * delta - 1.0, xi)
        return float(result)

    def scaling_dimension_from_mass(self, mass_sq_times_l_sq: float) -> float:
        d = 1.0
        discriminant = (d / 2.0)**2 + mass_sq_times_l_sq
        if discriminant < 0:
            raise ValueError("Below Breitenlohner-Freedman bound")
        return d / 2.0 + np.sqrt(discriminant)

    def breitenlohner_freedman_bound(self) -> float:
        d = 1.0
        return -(d**2) / 4.0

    def two_point_function(
        self, x: float, x_prime: float, delta: float
    ) -> float:
        sep = abs(x - x_prime)
        if sep < 1e-15:
            raise ValueError("Coincident points")
        from scipy.special import gamma
        C = (2.0**(2 * delta - 1) * gamma(delta + 0.5)) / (np.sqrt(np.pi) * gamma(delta))
        return C / sep**(2 * delta)

    def retarded_green_function(
        self, omega: float, q: float, delta: float
    ) -> complex:
        k = np.sqrt(abs(omega**2 - q**2) + 0j)
        nu = delta - 0.5
        if omega**2 > q**2:
            ratio = special.kv(nu, -1j * k * self.ads_radius) / special.kv(nu - 1, -1j * k * self.ads_radius)
        else:
            ratio = special.kv(nu, k * self.ads_radius) / special.kv(nu - 1, k * self.ads_radius)
        return -k * ratio

    def optical_conductivity_dc(self) -> float:
        e_sq_over_h = 3.87404e-5
        return e_sq_over_h * self.central_charge / 2.0

    def optical_conductivity(self, omega: float, omega_0: float = 1.0) -> complex:
        sigma_0 = self.optical_conductivity_dc()
        sigma_1 = sigma_0 * 0.1
        return sigma_0 + sigma_1 * (omega / omega_0) + 0j

    def wiedemann_franz_ratio(self) -> float:
        pi_sq_over_3 = np.pi**2 / 3.0
        kB = 1.380649e-23
        e = 1.602176634e-19
        L0 = pi_sq_over_3 * kB**2 / e**2
        return L0 * (1.0 - 3.0 / self.central_charge)

    def btz_black_hole_mass(self, temperature: float) -> float:
        kB = 1.380649e-23
        hbar = 1.054571817e-34
        beta = hbar / (kB * temperature)
        return (2.0 * np.pi * self.ads_radius / beta)**2

    def btz_hawking_temperature(self, mass: float) -> float:
        kB = 1.380649e-23
        hbar = 1.054571817e-34
        r_plus = self.ads_radius * np.sqrt(mass)
        return hbar * r_plus / (2.0 * np.pi * kB * self.ads_radius**2)

    def btz_entropy(self, temperature: float) -> float:
        kB = 1.380649e-23
        hbar = 1.054571817e-34
        r_plus = 2.0 * np.pi * kB * temperature * self.ads_radius**2 / hbar
        return 2.0 * np.pi * r_plus / (4.0 * self.newton_constant_3d)

    def lyapunov_exponent(self, temperature: float) -> float:
        kB = 1.380649e-23
        hbar = 1.054571817e-34
        bound = 2.0 * np.pi * kB * temperature / hbar
        return bound * (1.0 - 6.0 / self.central_charge**2)

    def scrambling_time(self, temperature: float, n_qubits: int) -> float:
        kB = 1.380649e-23
        hbar = 1.054571817e-34
        beta = hbar / (kB * temperature)
        return (beta / (2.0 * np.pi)) * np.log(n_qubits)

    def entanglement_entropy(self, ell: float, uv_cutoff: float) -> float:
        if ell <= uv_cutoff:
            raise ValueError("Interval length must exceed UV cutoff")
        return (self.central_charge / 3.0) * np.log(ell / uv_cutoff)

    def holographic_correction_to_entropy(
        self, ell: float, uv_cutoff: float, ir_cutoff: float
    ) -> float:
        base = self.entanglement_entropy(ell, uv_cutoff)
        correction = (self.central_charge / (12.0 * np.pi**2)) * np.log(ell / ir_cutoff)
        return base + correction

    def radial_cutoff_from_energy(self, energy_scale: float) -> float:
        hbar = 1.054571817e-34
        v_F = self.fermi_velocity
        return hbar * v_F / energy_scale

    def noise_screening_factor(
        self, system_size: float, coherence_length: float, delta_n: float
    ) -> float:
        ratio = coherence_length / system_size
        exponent = 4.0 * delta_n
        return ratio**exponent

    def holographic_coherence_enhancement(
        self, system_size: float, coherence_length: float
    ) -> float:
        ratio = system_size / coherence_length
        if ratio <= 0:
            raise ValueError("system_size must exceed coherence_length")
        return ratio**(self.central_charge / 6.0)