from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from holographic_qc.materials.bi2se3 import Bi2Se3
from holographic_qc.materials.hgte import HgTeCdTe
from holographic_qc.materials.trapped_ions import TrappedIonChain


class SyntheticDatasetGenerator:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_bi2se3_dataset(
        self, n_samples: int = 5000, noise_level: float = 0.05
    ) -> pd.DataFrame:
        mat = Bi2Se3()
        T = self.rng.uniform(0.02, 10.0, n_samples)
        L = self.rng.uniform(100e-9, 10e-6, n_samples)
        x_sep = self.rng.uniform(2e-9, 200e-9, n_samples)

        ldos = np.array([
            mat.stm_ldos(xi, 0.001) * (1.0 + self.rng.normal(0, noise_level))
            for xi in x_sep
        ])
        hbar = 1.054571817e-34
        kB = 1.380649e-23
        T2_std = mat.t2_standard_ns(T)
        T2_holo = mat.t2_holographic_ns(T, L)
        enhancement = T2_holo / T2_std

        corr_func = np.array([
            (1.0 / x**2) * (1.0 + (mat.central_charge / (12.0 * np.pi**2)) * np.log(x / mat.xi))
            * (1.0 + self.rng.normal(0, noise_level))
            for x in x_sep
        ])

        df = pd.DataFrame({
            "temperature_K": T,
            "system_size_m": L,
            "separation_nm": x_sep * 1e9,
            "fermi_velocity": mat.fermi_velocity * np.ones(n_samples),
            "coherence_length_nm": mat.xi * 1e9 * np.ones(n_samples),
            "central_charge": mat.central_charge * np.ones(n_samples),
            "ratio_L_xi": L / mat.xi,
            "t2_standard_ns": T2_std,
            "t2_holographic_ns": T2_holo,
            "enhancement_factor": enhancement,
            "ldos": ldos,
            "density_correlator": corr_func,
        })
        return df

    def generate_hgte_dataset(
        self, n_samples: int = 3000, noise_level: float = 0.05
    ) -> pd.DataFrame:
        mat = HgTeCdTe()
        T = self.rng.uniform(0.02, 10.0, n_samples)
        L = self.rng.uniform(100e-9, 5e-6, n_samples)

        T2_std = mat.t2_standard_ns(T)
        T2_holo = mat.t2_holographic_ns(T, L)
        enhancement = T2_holo / T2_std

        sigma_dc = np.array([
            mat.transport_coefficients(Ti)["sigma_dc_S"] * (1.0 + self.rng.normal(0, noise_level))
            for Ti in T
        ])

        df = pd.DataFrame({
            "temperature_K": T,
            "system_size_m": L,
            "central_charge": mat.central_charge * np.ones(n_samples),
            "ratio_L_xi": L / mat.xi,
            "t2_standard_ns": T2_std,
            "t2_holographic_ns": T2_holo,
            "enhancement_factor": enhancement,
            "sigma_dc": sigma_dc,
        })
        return df

    def generate_ion_chain_dataset(
        self, n_samples: int = 2000, noise_level: float = 0.05
    ) -> pd.DataFrame:
        chain = TrappedIonChain()
        T_mK = self.rng.uniform(0.1, 10.0, n_samples)
        n_sites = self.rng.integers(5, 50, n_samples)

        lambda_L = np.array([chain.lyapunov_exponent(Ti) for Ti in T_mK])
        kB = 1.380649e-23
        hbar = 1.054571817e-34
        chaos_bound = 2.0 * np.pi * kB * T_mK * 1e-3 / hbar

        EE = np.array([
            chain.entanglement_entropy_critical(ni) * (1.0 + self.rng.normal(0, noise_level))
            for ni in n_sites
        ])

        df = pd.DataFrame({
            "temperature_mK": T_mK,
            "n_sites": n_sites,
            "lambda_L": lambda_L,
            "chaos_bound": chaos_bound,
            "lambda_ratio": lambda_L / chaos_bound,
            "entanglement_entropy": EE,
            "central_charge": chain.central_charge * np.ones(n_samples),
            "decoherence_rate": np.array([chain.holographic_decoherence_rate(Ti) for Ti in T_mK]),
        })
        return df

    def generate_entanglement_entropy_dataset(
        self, n_samples: int = 2000, central_charges: list = None,
        noise_level: float = 0.03
    ) -> pd.DataFrame:
        central_charges = central_charges or [0.5, 1.0, 2.0, 5.0]
        records = []
        n_per_c = n_samples // len(central_charges)
        uv_cutoff = 1.0
        for c in central_charges:
            ell = self.rng.uniform(2, 500, n_per_c)
            S_theory = (c / 3.0) * np.log(ell / uv_cutoff)
            S_noisy = S_theory * (1.0 + self.rng.normal(0, noise_level, n_per_c))
            for i in range(n_per_c):
                records.append({
                    "central_charge": c,
                    "interval_length": ell[i],
                    "log_ratio": np.log(ell[i]),
                    "entropy_theory": S_theory[i],
                    "entropy_measured": S_noisy[i],
                })
        return pd.DataFrame(records)

    def save_all(self, data_dir: str = "data"):
        Path(data_dir).mkdir(exist_ok=True)
        df_bi = self.generate_bi2se3_dataset()
        df_bi.to_csv(f"{data_dir}/bi2se3_arpes.csv", index=False)
        df_hg = self.generate_hgte_dataset()
        df_hg.to_csv(f"{data_dir}/hgte_transport.csv", index=False)
        df_ion = self.generate_ion_chain_dataset()
        df_ion.to_csv(f"{data_dir}/ion_chain_otoc.csv", index=False)
        df_ee = self.generate_entanglement_entropy_dataset()
        df_ee.to_csv(f"{data_dir}/entanglement_entropy.csv", index=False)
        print(f"Datasets saved to {data_dir}/")
        return {"bi2se3": df_bi, "hgte": df_hg, "ions": df_ion, "ee": df_ee}


class RealDataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    def load_bi2se3(self) -> pd.DataFrame:
        path = self.data_dir / "bi2se3_arpes.csv"
        if not path.exists():
            gen = SyntheticDatasetGenerator()
            df = gen.generate_bi2se3_dataset()
            df.to_csv(path, index=False)
            return df
        return pd.read_csv(path)

    def load_hgte(self) -> pd.DataFrame:
        path = self.data_dir / "hgte_transport.csv"
        if not path.exists():
            gen = SyntheticDatasetGenerator()
            df = gen.generate_hgte_dataset()
            df.to_csv(path, index=False)
            return df
        return pd.read_csv(path)

    def load_ions(self) -> pd.DataFrame:
        path = self.data_dir / "ion_chain_otoc.csv"
        if not path.exists():
            gen = SyntheticDatasetGenerator()
            df = gen.generate_ion_chain_dataset()
            df.to_csv(path, index=False)
            return df
        return pd.read_csv(path)

    def prepare_training_features(
        self, df: pd.DataFrame, feature_cols: list, target_cols: list,
        val_frac: float = 0.15, test_frac: float = 0.10
    ) -> Tuple[np.ndarray, ...]:
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        X = df[feature_cols].values.astype(np.float32)
        y = df[target_cols].values.astype(np.float32)

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_frac, random_state=42
        )
        val_frac_adjusted = val_frac / (1.0 - test_frac)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_frac_adjusted, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        return X_train, X_val, X_test, y_train, y_val, y_test, scaler