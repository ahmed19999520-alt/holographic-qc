from __future__ import annotations

import numpy as np
import scipy.special as special
import scipy.integrate as integrate
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


@dataclass
class DilatonConfig:
    ads_radius: float
    mass_sq_times_l_sq: float = 0.0
    fermi_velocity: float = 5e5
    luttinger_parameter: float = 1.0


class DilatonField:
    def __init__(self, config: DilatonConfig):
        self.L = config.ads_radius
        self.m2L2 = config.mass_sq_times_l_sq
        self.v_F = config.fermi_velocity
        self.K = config.luttinger_parameter
        self.nu = np.sqrt(0.25 + self.m2L2)
        self.delta = 1.0 + self.nu

    def equation_of_motion_radial(
        self, z: float, phi: float, dphi: float, k_sq: float
    ) -> Tuple[float, float]:
        d2phi = (1.0 / z) * dphi + (k_sq + (self.nu**2 - 0.25) / z**2) * phi
        return dphi, d2phi

    def radial_solution_regular(self, z: float, k: float) -> complex:
        kz = abs(k) * z
        return z**0.5 * special.kv(self.nu, kz)

    def radial_solution_irregular(self, z: float, k: float) -> complex:
        kz = abs(k) * z
        return z**0.5 * special.iv(self.nu, kz)

    def fefferman_graham_expansion(
        self, z: float, phi0: float, phi1: float
    ) -> float:
        source_power = 1.0 - self.nu
        vev_power = 1.0 + self.nu
        return phi0 * z**source_power + phi1 * z**vev_power

    def on_shell_action(
        self, phi0_fn: Callable[[float], float],
        phi1_fn: Callable[[float], float],
        x_range: Tuple[float, float], n_points: int = 1000
    ) -> float:
        x = np.linspace(x_range[0], x_range[1], n_points)
        phi0 = np.array([phi0_fn(xi) for xi in x])
        phi1 = np.array([phi1_fn(xi) for xi in x])
        integrand = phi0 * phi1
        return float(integrate.trapezoid(integrand, x)) / (16.0 * np.pi * self.L)

    def two_point_function(self, x: float, delta: Optional[float] = None) -> float:
        if delta is None:
            delta = self.delta
        if abs(x) < 1e-15:
            raise ValueError("Coincident points")
        from scipy.special import gamma
        C = (2.0**(2 * delta - 1) * gamma(delta + 0.5)) / (np.sqrt(np.pi) * gamma(delta))
        return C / abs(x)**(2 * delta)

    def holographic_correlation_with_log_correction(
        self, x: float, xi: float, c: float, A: float = 1.0
    ) -> float:
        power_law = A / abs(x)**2
        log_correction = (c / (12.0 * np.pi**2)) * np.log(abs(x) / xi)
        return power_law * (1.0 + log_correction)

    def dynamical_structure_factor(
        self, q: float, omega: float, temperature: float, A: float = 1.0
    ) -> float:
        kB = 1.380649e-23
        hbar = 1.054571817e-34
        T_natural = kB * temperature / hbar
        threshold = self.v_F * abs(q)
        if omega <= threshold:
            return 0.0
        numerator = A / np.sqrt(omega**2 - threshold**2)
        bose = 1.0 / (1.0 - np.exp(-omega / T_natural))
        return numerator * bose

    def dynamical_structure_factor_holographic(
        self, q: float, omega: float, temperature: float,
        omega_uv: float, A: float = 1.0
    ) -> float:
        S0 = self.dynamical_structure_factor(q, omega, temperature, A)
        if self.m2L2 == 0.0:
            return S0
        log_corr = (self.m2L2 / (12.0 * np.pi)) * np.log(omega / omega_uv)
        return S0 * (1.0 + log_corr)

    def bosonization_density(self, phi: np.ndarray, dx: float) -> np.ndarray:
        return np.gradient(phi, dx) / np.pi

    def luttinger_liquid_propagator(
        self, x: float, t: float, temperature: float
    ) -> complex:
        kB = 1.380649e-23
        hbar = 1.054571817e-34
        beta = hbar / (kB * temperature)
        T_nat = 1.0 / beta
        arg_plus = np.pi * T_nat * (t + x / self.v_F + 1j * 0.0) + 1e-10j
        arg_minus = np.pi * T_nat * (t - x / self.v_F + 1j * 0.0) + 1e-10j
        G_plus = (np.pi * T_nat / np.sinh(arg_plus))**(1.0 / (2.0 * self.K))
        G_minus = (np.pi * T_nat / np.sinh(arg_minus))**(1.0 / (2.0 * self.K))
        return G_plus * G_minus

    def noise_power_spectrum(self, omega: float, gamma_0: float, temperature: float) -> float:
        kB = 1.380649e-23
        hbar = 1.054571817e-34
        bose = 1.0 / (np.exp(hbar * omega / (kB * temperature)) - 1.0)
        return gamma_0 * (bose + 0.5)

    def effective_noise_after_bulk_screening(
        self, system_size: float, coherence_length: float,
        delta_n: float, noise_power: float
    ) -> float:
        screening = (coherence_length / system_size)**(4.0 * delta_n)
        return noise_power * screening

    def optical_conductivity_edge(
        self, omega: float, sigma_0: float, omega_0: float
    ) -> complex:
        return sigma_0 * (1.0 + (omega / omega_0) * 0.1) + 0j

    def generate_field_configuration(
        self, z_values: np.ndarray, x: float, phi0: float
    ) -> np.ndarray:
        phi = np.zeros(len(z_values))
        for i, z in enumerate(z_values):
            phi[i] = phi0 * (z / self.L)**self.delta * np.exp(-abs(x) / self.L)
        return phi