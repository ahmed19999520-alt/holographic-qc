from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from holographic_qc.core.ryu_takayanagi import RyuTakayanagi, RTConfig


class StabilizerCode:
    def __init__(self, n: int, k: int, d: int):
        self.n = n
        self.k = k
        self.d = d
        self._stabilizers: List[np.ndarray] = []
        self._logicals_X: List[np.ndarray] = []
        self._logicals_Z: List[np.ndarray] = []

    def add_stabilizer(self, pauli_string: np.ndarray):
        self._stabilizers.append(pauli_string)

    def syndrome(self, error: np.ndarray) -> np.ndarray:
        synd = np.zeros(len(self._stabilizers), dtype=np.int8)
        for i, s in enumerate(self._stabilizers):
            synd[i] = int(np.sum(s * error) % 2)
        return synd

    def threshold_estimate(self) -> float:
        return 1.0 / (self.d * np.sqrt(2))


class SurfaceCode(StabilizerCode):
    def __init__(self, distance: int):
        n = distance**2 + (distance - 1)**2
        k = 1
        super().__init__(n, k, distance)
        self.distance = distance
        self._build_stabilizers()

    def _build_stabilizers(self):
        d = self.distance
        n = self.n
        for p in range((d - 1) * d):
            stab = np.zeros(2 * n, dtype=np.int8)
            stab[p] = 1
            self._stabilizers.append(stab)

    def holographic_code_distance(
        self, central_charge: float, system_size: float, lattice_spacing: float
    ) -> float:
        correction = (central_charge / (6.0 * np.pi)) * np.log(system_size / lattice_spacing)
        return self.distance * (1.0 + correction)

    def holographic_threshold(self, central_charge: float, p_std: float = 0.01) -> float:
        return p_std * (1.0 + central_charge / (12.0 * np.pi))

    def logical_error_rate(self, physical_error_rate: float) -> float:
        d = self.distance
        if physical_error_rate >= 0.5:
            return 0.5
        threshold = 0.01
        if physical_error_rate >= threshold:
            return 0.5
        ratio = physical_error_rate / threshold
        return 0.1 * ratio**(d // 2)


class PentagonHaPPYCode:
    def __init__(self, n_layers: int = 3):
        self.n_layers = n_layers
        self.n_boundary = 5 * 4**(n_layers - 1)
        self.n_bulk = (5 * (4**n_layers - 1)) // 3
        self.n_logical = self.n_bulk

    def encoding_rate(self) -> float:
        return self.n_logical / self.n_boundary

    def erasure_correction_radius(self, bulk_region_size: int) -> float:
        total_boundary = float(self.n_boundary)
        return float(bulk_region_size) / total_boundary

    def isometry_matrix(self) -> np.ndarray:
        n = self.n_boundary
        k = self.n_logical
        V = np.random.randn(n, k) + 1j * np.random.randn(n, k)
        Q, _ = np.linalg.qr(V)
        return Q[:, :k]

    def ryu_takayanagi_check(
        self, boundary_region: List[int], bulk_region: List[int],
        rt_config: RTConfig
    ) -> bool:
        rt = RyuTakayanagi(rt_config)
        boundary_size = len(boundary_region)
        complementary_size = self.n_boundary - boundary_size
        S_A = rt.entanglement_entropy_central_charge(float(boundary_size))
        S_Ac = rt.entanglement_entropy_central_charge(float(complementary_size))
        return S_A >= len(bulk_region)

    def decode(self, syndrome: np.ndarray, error_rate: float) -> np.ndarray:
        correction = np.zeros(self.n_boundary, dtype=np.int8)
        for i, s in enumerate(syndrome):
            if s == 1:
                correction[i % self.n_boundary] = 1
        return correction


class HolographicCode:
    def __init__(
        self, code_type: str = "pentagon",
        central_charge: float = 1.0,
        rt_config: Optional[RTConfig] = None
    ):
        self.code_type = code_type
        self.c = central_charge
        self.rt_config = rt_config
        if code_type == "pentagon":
            self.base_code = PentagonHaPPYCode()
        elif code_type == "surface":
            self.base_code = SurfaceCode(distance=7)
        else:
            raise ValueError(f"Unknown code type: {code_type}")

    def encoding_rate(self) -> float:
        return self.base_code.encoding_rate() if hasattr(self.base_code, 'encoding_rate') else 1.0 / self.c

    def effective_distance(self, system_size: float, lattice_spacing: float) -> float:
        if isinstance(self.base_code, SurfaceCode):
            return self.base_code.holographic_code_distance(self.c, system_size, lattice_spacing)
        return float(self.c) * np.log(system_size / lattice_spacing)

    def error_threshold(self, p_std: float = 0.01) -> float:
        if isinstance(self.base_code, SurfaceCode):
            return self.base_code.holographic_threshold(self.c, p_std)
        return p_std * (1.0 + self.c / (12.0 * np.pi))

    def logical_error_rate(
        self, physical_rate: float, system_size: float,
        lattice_spacing: float, n_levels: int = 5
    ) -> float:
        d_eff = self.effective_distance(system_size, lattice_spacing)
        p_thresh = self.error_threshold()
        if physical_rate >= p_thresh:
            return 0.5
        ratio = physical_rate / p_thresh
        return 0.1 * ratio**(int(d_eff) // 2)

    def quantum_memory_time(
        self, T2_physical: float, system_size: float, coherence_length: float
    ) -> float:
        k = self.encoding_rate()
        c = self.c
        suppression = np.exp(-k * c * np.log(system_size / coherence_length) / 6.0)
        return T2_physical / suppression

    def resource_overhead(self, n_logical: int) -> dict:
        if isinstance(self.base_code, PentagonHaPPYCode):
            rate = self.base_code.encoding_rate()
            n_phys = int(np.ceil(n_logical / rate))
        else:
            n_phys = n_logical * int(self.c * 10)
        return {
            "n_logical_qubits": n_logical,
            "n_physical_qubits": n_phys,
            "overhead_ratio": n_phys / n_logical,
            "encoding_rate": self.encoding_rate(),
        }