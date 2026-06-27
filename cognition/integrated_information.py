from __future__ import annotations

import numpy as np
import itertools
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
from scipy.special import rel_entr
from scipy.stats import entropy as scipy_entropy


@dataclass
class IITConfig:
    n_elements: int = 8
    connectivity_density: float = 0.5
    noise_level: float = 0.01
    n_states: int = None
    phi_approximation: str = "exact"

    def __post_init__(self):
        if self.n_states is None:
            self.n_states = 2**self.n_elements


@dataclass
class CauseEffectStructure:
    mechanism: FrozenSet[int]
    purview: FrozenSet[int]
    phi: float
    cause_repertoire: np.ndarray
    effect_repertoire: np.ndarray
    is_core: bool = False


class IntegratedInformationTheory:
    def __init__(self, config: IITConfig):
        self.cfg = config
        self.n = config.n_elements
        self.N = 2**self.n
        self._W: Optional[np.ndarray] = None
        self._tpm: Optional[np.ndarray] = None
        self._ces: List[CauseEffectStructure] = []

    def set_connectivity(self, W: np.ndarray):
        assert W.shape == (self.n, self.n), f"W must be {self.n}x{self.n}"
        self._W = W.copy()
        self._tpm = self._build_tpm(W)

    def random_connectivity(self, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        W = rng.uniform(-1, 1, (self.n, self.n))
        mask = rng.random((self.n, self.n)) < self.cfg.connectivity_density
        W = W * mask
        np.fill_diagonal(W, 0.0)
        self.set_connectivity(W)
        return W

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _state_to_vec(self, state_idx: int) -> np.ndarray:
        return np.array([(state_idx >> i) & 1 for i in range(self.n)], dtype=np.float64)

    def _vec_to_state(self, vec: np.ndarray) -> int:
        return int(sum(int(round(v)) * (2**i) for i, v in enumerate(vec)))

    def _build_tpm(self, W: np.ndarray) -> np.ndarray:
        tpm = np.zeros((self.N, self.N))
        for s in range(self.N):
            x = self._state_to_vec(s)
            probs = self._sigmoid(W @ x + self.cfg.noise_level * np.random.randn(self.n))
            probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
            for s_next in range(self.N):
                x_next = self._state_to_vec(s_next)
                p = np.prod([
                    probs[i] if x_next[i] == 1 else 1.0 - probs[i]
                    for i in range(self.n)
                ])
                tpm[s, s_next] = p
        row_sums = tpm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return tpm / row_sums

    def _marginal_tpm(self, tpm: np.ndarray, subset: List[int]) -> np.ndarray:
        n_sub = len(subset)
        N_sub = 2**n_sub
        complement = [i for i in range(self.n) if i not in subset]
        m_tpm = np.zeros((N_sub, N_sub))
        for s in range(self.N):
            s_bits = self._state_to_vec(s)
            s_sub_idx = int(sum(int(s_bits[subset[k]]) * 2**k for k in range(n_sub)))
            for s_next in range(self.N):
                s_next_bits = self._state_to_vec(s_next)
                s_next_sub_idx = int(sum(int(s_next_bits[subset[k]]) * 2**k for k in range(n_sub)))
                m_tpm[s_sub_idx, s_next_sub_idx] += tpm[s, s_next] / (2**len(complement))
        return m_tpm

    def cause_repertoire(self, mechanism: List[int], purview: List[int], state: int) -> np.ndarray:
        n_purview = len(purview)
        N_purview = 2**n_purview
        cause_rep = np.zeros(N_purview)
        mech_state = self._state_to_vec(state)
        for p_s in range(N_purview):
            p_bits = self._state_to_vec(p_s) if n_purview <= self.n else np.zeros(n_purview)
            p = 1.0
            for i, pi in enumerate(purview):
                if pi >= self.n:
                    continue
                if n_purview > 0 and i < len(p_bits):
                    x_pi = p_bits[i] if i < n_purview else 0.0
                else:
                    x_pi = 0.5
                for mi, m_idx in enumerate(mechanism):
                    if m_idx >= self.n:
                        continue
                    x_m = mech_state[m_idx]
                    w = self._W[m_idx, pi] if self._W is not None else 0.0
                    activation = self._sigmoid(w * x_pi)
                    p *= activation if x_m > 0.5 else (1.0 - activation)
            cause_rep[p_s] = max(p, 1e-10)
        total = cause_rep.sum()
        return cause_rep / max(total, 1e-10)

    def effect_repertoire(self, mechanism: List[int], purview: List[int], state: int) -> np.ndarray:
        if self._tpm is None:
            raise RuntimeError("TPM not built. Call set_connectivity() first.")
        n_purview = len(purview)
        N_purview = 2**n_purview
        effect_rep = np.zeros(N_purview)
        mech_state_vec = self._state_to_vec(state)
        for p_s in range(N_purview):
            p_bits = self._state_to_vec(p_s) if p_s < self.N else np.zeros(n_purview)
            p = 1.0
            for i, pi in enumerate(purview):
                if pi >= self.n:
                    continue
                probs_pi = 0.0
                count = 0
                for full_s in range(self.N):
                    s_vec = self._state_to_vec(full_s)
                    mech_match = all(
                        abs(s_vec[m] - mech_state_vec[m]) < 0.5
                        for m in mechanism if m < self.n
                    )
                    if mech_match:
                        probs_pi += self._tpm[full_s, p_s % self.N] if p_s < self.N else 0.5
                        count += 1
                avg_prob = probs_pi / max(count, 1)
                target_bit = p_bits[i] if i < n_purview else 0.0
                p *= avg_prob if target_bit > 0.5 else (1.0 - avg_prob)
            effect_rep[p_s] = max(p, 1e-10)
        total = effect_rep.sum()
        return effect_rep / max(total, 1e-10)

    def phi_mechanism(self, mechanism: List[int], purview: List[int], state: int) -> float:
        if len(mechanism) == 0 or len(purview) == 0:
            return 0.0
        cause_full = self.cause_repertoire(mechanism, purview, state)
        effect_full = self.effect_repertoire(mechanism, purview, state)
        n_purview = len(purview)
        N_purview = 2**n_purview
        uniform = np.ones(N_purview) / N_purview
        kl_cause = float(np.sum(rel_entr(cause_full + 1e-10, uniform + 1e-10)))
        kl_effect = float(np.sum(rel_entr(effect_full + 1e-10, uniform + 1e-10)))
        phi_min = min(kl_cause, kl_effect)
        return max(0.0, phi_min)

    def phi_system(self, state: int, max_mechanisms: int = 16) -> float:
        if self._W is None:
            raise RuntimeError("Connectivity not set.")
        total_phi = 0.0
        elements = list(range(min(self.n, 6)))
        for r in range(1, min(len(elements) + 1, 4)):
            for mech in itertools.combinations(elements, r):
                for purview_size in range(1, min(len(elements) + 1, 4)):
                    for purview in itertools.combinations(elements, purview_size):
                        phi = self.phi_mechanism(list(mech), list(purview), state)
                        total_phi += phi
        return total_phi

    def phi_approximation_lz(self, state: int) -> float:
        import zlib
        if self._tpm is None:
            return 0.0
        state_vec = self._state_to_vec(state)
        tpm_row = self._tpm[state]
        data = state_vec.tobytes() + tpm_row.tobytes()
        compressed_len = len(zlib.compress(data, level=9))
        original_len = len(data)
        complexity = 1.0 - compressed_len / original_len
        return float(np.clip(complexity * self.n, 0.0, self.n))

    def compute_phi(self, state: Optional[int] = None) -> float:
        if state is None:
            state = np.random.randint(0, self.N)
        if self.cfg.phi_approximation == "lz":
            return self.phi_approximation_lz(state)
        return self.phi_system(state)

    def cause_effect_structure(self, state: int) -> List[CauseEffectStructure]:
        ces = []
        elements = list(range(min(self.n, 5)))
        for r in range(1, min(4, len(elements) + 1)):
            for mech in itertools.combinations(elements, r):
                best_phi = 0.0
                best_purview = tuple(elements[:1])
                best_cause = np.array([0.5, 0.5])
                best_effect = np.array([0.5, 0.5])
                for purview_size in range(1, min(4, len(elements) + 1)):
                    for purview in itertools.combinations(elements, purview_size):
                        phi = self.phi_mechanism(list(mech), list(purview), state)
                        if phi > best_phi:
                            best_phi = phi
                            best_purview = purview
                            best_cause = self.cause_repertoire(list(mech), list(purview), state)
                            best_effect = self.effect_repertoire(list(mech), list(purview), state)
                if best_phi > 0.0:
                    ces.append(CauseEffectStructure(
                        mechanism=frozenset(mech),
                        purview=frozenset(best_purview),
                        phi=best_phi,
                        cause_repertoire=best_cause,
                        effect_repertoire=best_effect,
                    ))
        self._ces = ces
        return ces

    def conceptual_information(self, state: int) -> float:
        ces = self.cause_effect_structure(state)
        return sum(c.phi for c in ces)

    def big_phi(self, state: int) -> float:
        phi_sys = self.compute_phi(state)
        return phi_sys

    def consciousness_level(self, state: int) -> dict:
        phi = self.big_phi(state)
        phi_norm = phi / max(self.n, 1)
        return {
            "phi": phi,
            "phi_normalized": phi_norm,
            "n_elements": self.n,
            "state": state,
            "level": "high" if phi_norm > 0.3 else "medium" if phi_norm > 0.1 else "low",
        }