from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class HgTeCdTe:
    fermi_velocity: float = 3.0e5
    bulk_gap_eV: float = 0.010
    lattice_constant_nm: float = 0.646
    well_width_nm: float = 7.0
    n_edge_channels: int = 2
    phonon_coupling_alpha: float = 1.8e-2

    def __post_init__(self):
        hbar = 1.054571817e-34
        eV = 1.602176634e-19
        self.xi = hbar * self.fermi_velocity / (self.bulk_gap_eV * eV)
        self.central_charge = float(self.n_edge_channels)

    @property
    def xi_nm(self) -> float:
        return self.xi * 1e9

    def gap_as_function_of_well_width(self, d_nm: float) -> float:
        d_c = 6.3
        return self.bulk_gap_eV * np.tanh((d_nm - d_c) / 0.5)

    def t2_standard_ns(self, temperature_K: float) -> float:
        hbar = 1.054571817e-34
        kB = 1.380649e-23
        rate = self.phonon_coupling_alpha * (kB * temperature_K / hbar)**2
        return 1e9 / rate

    def t2_holographic_ns(self, temperature_K: float, system_size_m: float) -> float:
        T2_std = self.t2_standard_ns(temperature_K) * 1e-9
        ratio = system_size_m / self.xi
        enhancement = ratio**(self.central_charge / 6.0)
        return T2_std * enhancement * 1e9

    def transport_coefficients(self, temperature_K: float) -> dict:
        kB = 1.380649e-23
        e = 1.602176634e-19
        hbar = 1.054571817e-34
        L0 = (np.pi**2 / 3.0) * kB**2 / e**2
        sigma_0 = (e**2 / (2.0 * np.pi * hbar)) * self.central_charge
        kappa_over_sigmaT = L0 * (1.0 - 3.0 / self.central_charge)
        return {
            "sigma_dc_S": sigma_0,
            "lorenz_ratio": kappa_over_sigmaT,
            "WF_violation_fraction": 3.0 / self.central_charge,
        }

    def material_parameters_dict(self) -> dict:
        return {
            "fermi_velocity_m_s": self.fermi_velocity,
            "bulk_gap_eV": self.bulk_gap_eV,
            "coherence_length_nm": self.xi * 1e9,
            "central_charge": self.central_charge,
        }