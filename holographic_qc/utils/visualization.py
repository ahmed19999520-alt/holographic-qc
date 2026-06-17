from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class HolographicVisualizer:
    def __init__(self, output_dir: str = "figures", dpi: int = 150):
        self.out = Path(output_dir)
        self.out.mkdir(exist_ok=True)
        self.dpi = dpi
        plt.rcParams.update({
            "font.size": 12,
            "axes.labelsize": 13,
            "legend.fontsize": 10,
            "figure.dpi": dpi,
        })

    def plot_entanglement_entropy_scaling(
        self,
        ell_values: np.ndarray,
        S_measured: np.ndarray,
        central_charges: List[float],
        uv_cutoff: float = 1.0,
        save: bool = True,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 5))
        log_ell = np.log(ell_values / uv_cutoff)
        ax.scatter(log_ell, S_measured, color="black", s=20, zorder=5, label="Measured")
        colors = ["blue", "red", "green", "purple", "orange"]
        for ci, c in enumerate(central_charges):
            S_theory = (c / 3.0) * log_ell
            ax.plot(log_ell, S_theory, color=colors[ci % len(colors)],
                    linewidth=2, label=f"$c={c}$: slope $c/3={c/3:.3f}$")
        ax.set_xlabel(r"$\ln(\ell/a)$")
        ax.set_ylabel(r"$S_A$")
        ax.set_title("Entanglement Entropy Scaling: CFT$_2$ Signature")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if save:
            fig.savefig(self.out / "entanglement_entropy_scaling.pdf", bbox_inches="tight")
        return fig

    def plot_coherence_enhancement(
        self,
        L_over_xi: np.ndarray,
        central_charges: List[float],
        save: bool = True,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["blue", "red", "green", "purple", "orange"]
        for ci, c in enumerate(central_charges):
            enh = L_over_xi**(c / 6.0)
            ax.semilogy(L_over_xi, enh, color=colors[ci % len(colors)],
                        linewidth=2.5, label=f"$c={c}$: $(L/\\xi)^{{{c/6:.2f}}}$")
        ax.axhline(1000, color="orange", linestyle="--", linewidth=1.5, label="Fault-tolerance threshold")
        ax.set_xlabel(r"$L/\xi$")
        ax.set_ylabel(r"$T_2^{\rm holo}/T_2^{\rm std}$")
        ax.set_title("Holographic Coherence Enhancement")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        if save:
            fig.savefig(self.out / "coherence_enhancement.pdf", bbox_inches="tight")
        return fig

    def plot_density_correlations(
        self,
        r_values: np.ndarray,
        C_measured: np.ndarray,
        C_cft: np.ndarray,
        C_holo: np.ndarray,
        save: bool = True,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.loglog(r_values, np.abs(C_measured), "ko", markersize=4, label="STM data")
        ax.loglog(r_values, np.abs(C_cft), "b-", linewidth=2, label="Pure CFT: $|x|^{-2\\Delta}$")
        ax.loglog(r_values, np.abs(C_holo), "r--", linewidth=2, label="Holographic correction")
        ax.set_xlabel(r"$|x - x'|$ [nm]")
        ax.set_ylabel(r"$C(x, x')$ [arb. u.]")
        ax.set_title("Density-Density Correlations: CFT vs. Holographic")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        if save:
            fig.savefig(self.out / "density_correlations.pdf", bbox_inches="tight")
        return fig

    def plot_otoc(
        self,
        times: np.ndarray,
        F_holographic: np.ndarray,
        F_integrable: np.ndarray,
        F_generic: np.ndarray,
        save: bool = True,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(times, F_holographic, "r-", linewidth=2.5, label="Holographic: $e^{-\\lambda_L t}$")
        ax.semilogy(times, np.abs(F_integrable), "b--", linewidth=2, label="Integrable: oscillatory")
        ax.semilogy(times, F_generic, "g:", linewidth=2, label="Generic chaotic")
        ax.axhline(np.exp(-1), color="purple", linestyle="-.", linewidth=1.5, label="Chaos bound level")
        ax.set_xlabel(r"$t \lambda_L$")
        ax.set_ylabel(r"$F(t) = -\langle[W(t), V(0)]^2\rangle$")
        ax.set_title("Out-of-Time-Order Correlator: Holographic Scrambling")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        if save:
            fig.savefig(self.out / "otoc.pdf", bbox_inches="tight")
        return fig

    def plot_training_history(
        self,
        train_losses: List[float],
        val_losses: List[float],
        title: str = "Training History",
        save: bool = True,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 4))
        epochs = np.arange(len(train_losses))
        ax.semilogy(epochs, train_losses, "b-", linewidth=2, label="Train loss")
        ax.semilogy(epochs, val_losses, "r-", linewidth=2, label="Validation loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        if save:
            fig.savefig(self.out / "training_history.pdf", bbox_inches="tight")
        return fig

    def plot_grover_amplitudes(
        self,
        n_qubits: int,
        target_states: List[int],
        n_iterations_range: range = None,
        save: bool = True,
    ) -> plt.Figure:
        from holographic_qc.algorithms.grover import GroverAlgorithm
        n_iter_range = n_iterations_range or range(0, 15)
        success_probs = []
        for k in n_iter_range:
            grover = GroverAlgorithm(n_qubits, target_states)
            success_probs.append(grover.success_probability(k))
        optimal = GroverAlgorithm(n_qubits, target_states).optimal_iterations()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(list(n_iter_range), success_probs, "b-o", markersize=5, linewidth=2)
        ax.axvline(optimal, color="red", linestyle="--", label=f"Optimal: $k={optimal}$")
        ax.set_xlabel("Number of Grover iterations $k$")
        ax.set_ylabel("Success probability $P_{\\rm success}$")
        ax.set_title(f"Grover Search: $n={n_qubits}$ qubits, {len(target_states)} target(s)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if save:
            fig.savefig(self.out / "grover_amplitudes.pdf", bbox_inches="tight")
        return fig

    def plot_shor_qft_spectrum(
        self,
        a: int, N: int, n_precision: int = 8,
        save: bool = True,
    ) -> plt.Figure:
        from holographic_qc.algorithms.shor import PeriodFinder
        import math
        finder = PeriodFinder(a, N, n_precision)
        state = finder.initial_state()
        state = finder.apply_modular_exp(state)
        from holographic_qc.algorithms.qft import QuantumFourierTransform
        qft = QuantumFourierTransform(n_precision)
        freq_state = qft.apply(state)
        probs = np.abs(freq_state)**2
        dim = 2**n_precision
        freqs = np.arange(dim) / dim
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(np.arange(len(state)), np.abs(state)**2, color="blue", alpha=0.7)
        axes[0].set_title(f"Modular exponent state: $a={a}$, $N={N}$")
        axes[0].set_xlabel("Basis state index")
        axes[0].set_ylabel("Probability")
        axes[1].bar(freqs, probs, color="red", alpha=0.7, width=1.0 / dim)
        axes[1].set_title("QFT output: Period peaks")
        axes[1].set_xlabel("Frequency $j/2^n$")
        axes[1].set_ylabel("Probability")
        plt.tight_layout()
        if save:
            fig.savefig(self.out / f"shor_qft_a{a}_N{N}.pdf", bbox_inches="tight")
        return fig

    def plot_materials_comparison(
        self,
        temperatures: np.ndarray,
        save: bool = True,
    ) -> plt.Figure:
        from holographic_qc.materials.bi2se3 import Bi2Se3
        from holographic_qc.materials.hgte import HgTeCdTe
        bi = Bi2Se3()
        hg = HgTeCdTe()
        T2_bi_std = np.array([bi.t2_standard_ns(T) for T in temperatures])
        T2_bi_holo = np.array([bi.t2_holographic_ns(T, 1e-6) for T in temperatures])
        T2_hg_std = np.array([hg.t2_standard_ns(T) for T in temperatures])
        T2_hg_holo = np.array([hg.t2_holographic_ns(T, 1e-6) for T in temperatures])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.loglog(temperatures, T2_bi_std, "b-", linewidth=2, label=r"Bi$_2$Se$_3$ standard")
        ax.loglog(temperatures, T2_bi_holo, "b--", linewidth=2, label=r"Bi$_2$Se$_3$ holographic")
        ax.loglog(temperatures, T2_hg_std, "r-", linewidth=2, label="HgTe/CdTe standard")
        ax.loglog(temperatures, T2_hg_holo, "r--", linewidth=2, label="HgTe/CdTe holographic")
        ax.axhline(1.0, color="green", linestyle=":", linewidth=1.5, label="$T_2 = 1$ ns")
        ax.set_xlabel("Temperature $T$ [K]")
        ax.set_ylabel("$T_2$ [ns]")
        ax.set_title("Coherence Times: Standard vs. Holographic")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        if save:
            fig.savefig(self.out / "materials_comparison.pdf", bbox_inches="tight")
        return fig

    def plot_vqe_convergence(
        self,
        energy_history: List[float],
        exact_energy: float,
        save: bool = True,
    ) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        iters = np.arange(len(energy_history))
        axes[0].plot(iters, energy_history, "b-", linewidth=1.5)
        axes[0].axhline(exact_energy, color="red", linestyle="--", linewidth=2, label=f"Exact: {exact_energy:.4f}")
        axes[0].set_xlabel("Optimizer iteration")
        axes[0].set_ylabel("Energy [J]")
        axes[0].set_title("VQE Energy Convergence")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        errors = np.abs(np.array(energy_history) - exact_energy)
        errors = np.maximum(errors, 1e-20)
        axes[1].semilogy(iters, errors, "r-", linewidth=1.5)
        axes[1].set_xlabel("Optimizer iteration")
        axes[1].set_ylabel(r"$|E - E_{\rm exact}|$")
        axes[1].set_title("VQE Energy Error")
        axes[1].grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        if save:
            fig.savefig(self.out / "vqe_convergence.pdf", bbox_inches="tight")
        return fig

    def generate_all_plots(self):
        print("Generating all figures...")
        L_xi = np.linspace(1, 50, 200)
        self.plot_coherence_enhancement(L_xi, [0.5, 1.0, 2.0, 5.0])
        temperatures = np.logspace(-2, 1, 100)
        self.plot_materials_comparison(temperatures)
        t = np.linspace(0, 5, 200)
        F_holo = np.exp(-2.0 * t)
        F_int = 0.5 + 0.5 * np.cos(2.0 * t)
        F_gen = np.exp(-1.2 * t)
        self.plot_otoc(t, F_holo, F_int, F_gen)
        self.plot_grover_amplitudes(6, [5])
        self.plot_shor_qft_spectrum(7, 15)
        print(f"All figures saved to {self.out}/")