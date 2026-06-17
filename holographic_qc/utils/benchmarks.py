from __future__ import annotations

import time
import numpy as np
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    name: str
    elapsed_seconds: float
    n_calls: int
    output: object = None
    metadata: Dict = field(default_factory=dict)

    @property
    def calls_per_second(self) -> float:
        return self.n_calls / max(self.elapsed_seconds, 1e-12)


class Benchmarker:
    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def run(
        self, name: str, fn: Callable, args: tuple = (), kwargs: dict = None,
        n_calls: int = 1, warmup: int = 0
    ) -> BenchmarkResult:
        kwargs = kwargs or {}
        for _ in range(warmup):
            fn(*args, **kwargs)
        t0 = time.perf_counter()
        output = None
        for _ in range(n_calls):
            output = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        result = BenchmarkResult(name=name, elapsed_seconds=elapsed, n_calls=n_calls, output=output)
        self.results.append(result)
        return result

    def print_report(self):
        print(f"\n{'=' * 65}")
        print(f"{'Benchmark':40} | {'Time [ms]':>10} | {'Calls/s':>12}")
        print(f"{'-' * 65}")
        for r in self.results:
            ms = r.elapsed_seconds * 1000 / r.n_calls
            cps = r.calls_per_second
            print(f"{r.name:40} | {ms:10.3f} | {cps:12.1f}")
        print(f"{'=' * 65}\n")

    def benchmark_virasoro(self, n_modes: int = 10, n_calls: int = 100) -> BenchmarkResult:
        from holographic_qc.core.virasoro import VirasoroAlgebra, VirasoroConfig
        alg = VirasoroAlgebra(VirasoroConfig(central_charge=1.0, max_mode=n_modes))
        return self.run(
            f"VirasoroAlgebra.commutator (N={n_modes})",
            alg.commutator_scalar, args=(3, -3), n_calls=n_calls
        )

    def benchmark_qft(self, n_qubits: int = 8, n_calls: int = 10) -> BenchmarkResult:
        from holographic_qc.algorithms.qft import QuantumFourierTransform
        qft = QuantumFourierTransform(n_qubits)
        state = np.random.randn(2**n_qubits) + 1j * np.random.randn(2**n_qubits)
        state /= np.linalg.norm(state)
        return self.run(
            f"QFT.apply (n={n_qubits})",
            qft.apply, args=(state,), n_calls=n_calls
        )

    def benchmark_grover(self, n_qubits: int = 8, n_calls: int = 5) -> BenchmarkResult:
        from holographic_qc.algorithms.grover import GroverAlgorithm
        grover = GroverAlgorithm(n_qubits, [42])
        return self.run(
            f"Grover.run (n={n_qubits})",
            grover.run, n_calls=n_calls
        )

    def benchmark_shor(self, N: int = 15, n_calls: int = 20) -> BenchmarkResult:
        from holographic_qc.algorithms.shor import ShorAlgorithm
        shor = ShorAlgorithm(N)
        return self.run(
            f"Shor.factor_classical (N={N})",
            shor.factor_classical_simulation, n_calls=n_calls
        )

    def benchmark_decoherence(self, n_calls: int = 100) -> BenchmarkResult:
        from holographic_qc.protection.decoherence import HolographicDecoherence
        from holographic_qc.core.ads_cft import AdsCft3
        from holographic_qc.materials.bi2se3 import Bi2Se3
        mat = Bi2Se3()
        ads = AdsCft3(central_charge=mat.central_charge, ads_radius=mat.xi)
        dec = HolographicDecoherence(ads, mat)
        return self.run(
            "HolographicDecoherence.coherence_time_ratio",
            dec.coherence_time_ratio, args=(1e-6,), n_calls=n_calls
        )

    def benchmark_rt(self, n_calls: int = 500) -> BenchmarkResult:
        from holographic_qc.core.ryu_takayanagi import RyuTakayanagi, RTConfig
        config = RTConfig(central_charge=1.0, newton_constant_3d=1.1e-9, ads_radius=1.1e-9, uv_cutoff=0.3e-9)
        rt = RyuTakayanagi(config)
        return self.run(
            "RyuTakayanagi.entanglement_entropy",
            rt.entanglement_entropy_central_charge, args=(100e-9,), n_calls=n_calls
        )

    def run_all(self) -> List[BenchmarkResult]:
        self.benchmark_virasoro()
        self.benchmark_rt()
        self.benchmark_decoherence()
        self.benchmark_qft(n_qubits=6)
        self.benchmark_qft(n_qubits=8)
        self.benchmark_grover(n_qubits=6)
        self.benchmark_grover(n_qubits=8)
        self.benchmark_shor(N=15)
        self.benchmark_shor(N=35)
        self.print_report()
        return self.results