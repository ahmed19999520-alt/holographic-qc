from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import deque
import threading


@dataclass
class WorkspaceConfig:
    capacity: int = 7
    broadcast_threshold: float = 0.6
    competition_tau: float = 0.1
    decay_rate: float = 0.05
    integration_window_ms: float = 200.0
    n_specialists: int = 8
    workspace_dim: int = 512


@dataclass
class BroadcastMessage:
    content: np.ndarray
    source_id: str
    salience: float
    timestamp: float
    message_type: str = "percept"
    metadata: Dict = field(default_factory=dict)

    def age_ms(self) -> float:
        return (time.perf_counter() - self.timestamp) * 1000.0


class Specialist:
    def __init__(
        self,
        specialist_id: str,
        process_fn: Callable[[np.ndarray], Tuple[np.ndarray, float]],
        domain: str = "generic",
        capacity: int = 3,
    ):
        self.id = specialist_id
        self.domain = domain
        self.capacity = capacity
        self._process = process_fn
        self.activation: float = 0.0
        self.buffer: deque = deque(maxlen=capacity)
        self.n_broadcasts_received: int = 0
        self.n_broadcasts_sent: int = 0

    def receive_broadcast(self, message: BroadcastMessage):
        self.buffer.append(message)
        self.n_broadcasts_received += 1
        self.activation = min(1.0, self.activation + 0.1)

    def process(self, stimulus: np.ndarray) -> Tuple[np.ndarray, float]:
        output, salience = self._process(stimulus)
        self.n_broadcasts_sent += 1
        return output, salience

    def decay(self, rate: float = 0.05):
        self.activation = max(0.0, self.activation - rate)

    def state_dict(self) -> dict:
        return {
            "id": self.id,
            "domain": self.domain,
            "activation": self.activation,
            "buffer_size": len(self.buffer),
            "n_recv": self.n_broadcasts_received,
            "n_sent": self.n_broadcasts_sent,
        }


class GlobalWorkspace:
    def __init__(self, config: WorkspaceConfig):
        self.cfg = config
        self.specialists: Dict[str, Specialist] = {}
        self._workspace: List[BroadcastMessage] = []
        self._attention_weights: np.ndarray = np.zeros(config.n_specialists)
        self._broadcast_history: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
        self._cycle_count: int = 0
        self._global_state: np.ndarray = np.zeros(config.workspace_dim)

    def register_specialist(self, specialist: Specialist):
        self.specialists[specialist.id] = specialist

    def _competition(self, candidates: List[BroadcastMessage]) -> Optional[BroadcastMessage]:
        if not candidates:
            return None
        saliences = np.array([m.salience for m in candidates])
        ages = np.array([m.age_ms() for m in candidates])
        window = self.cfg.integration_window_ms
        age_penalties = np.exp(-ages / window)
        scores = saliences * age_penalties
        winner_idx = int(np.argmax(scores))
        if scores[winner_idx] >= self.cfg.broadcast_threshold:
            return candidates[winner_idx]
        return None

    def _broadcast(self, message: BroadcastMessage):
        for sp in self.specialists.values():
            sp.receive_broadcast(message)
        self._broadcast_history.append(message)
        alpha = 0.3
        if message.content.shape == self._global_state.shape:
            self._global_state = (1 - alpha) * self._global_state + alpha * message.content
        elif len(message.content) <= len(self._global_state):
            self._global_state[:len(message.content)] = (
                (1 - alpha) * self._global_state[:len(message.content)]
                + alpha * message.content
            )

    def step(self, stimuli: Dict[str, np.ndarray]) -> Optional[BroadcastMessage]:
        with self._lock:
            self._cycle_count += 1
            candidates: List[BroadcastMessage] = []
            for sp_id, stimulus in stimuli.items():
                if sp_id not in self.specialists:
                    continue
                sp = self.specialists[sp_id]
                output, salience = sp.process(stimulus)
                msg = BroadcastMessage(
                    content=output,
                    source_id=sp_id,
                    salience=salience,
                    timestamp=time.perf_counter(),
                    message_type="specialist_output",
                )
                candidates.append(msg)
            winner = self._competition(candidates)
            if winner is not None:
                self._broadcast(winner)
                self._workspace.append(winner)
                if len(self._workspace) > self.cfg.capacity:
                    self._workspace.pop(0)
            for sp in self.specialists.values():
                sp.decay(self.cfg.decay_rate)
            return winner

    def global_state(self) -> np.ndarray:
        return self._global_state.copy()

    def workspace_contents(self) -> List[BroadcastMessage]:
        return list(self._workspace)

    def attention_map(self) -> Dict[str, float]:
        total = sum(sp.activation for sp in self.specialists.values())
        if total < 1e-10:
            return {k: 0.0 for k in self.specialists}
        return {k: sp.activation / total for k, sp in self.specialists.items()}

    def conscious_access_report(self) -> dict:
        return {
            "cycle": self._cycle_count,
            "workspace_size": len(self._workspace),
            "global_state_norm": float(np.linalg.norm(self._global_state)),
            "attention_map": self.attention_map(),
            "n_total_broadcasts": len(self._broadcast_history),
            "active_specialists": sum(
                1 for sp in self.specialists.values() if sp.activation > 0.1
            ),
        }

    def ignition_score(self) -> float:
        if not self._broadcast_history:
            return 0.0
        recent = [m for m in self._broadcast_history if m.age_ms() < 500.0]
        if not recent:
            return 0.0
        saliences = [m.salience for m in recent]
        return float(np.mean(saliences))

    def create_specialist(
        self,
        specialist_id: str,
        weight_matrix: np.ndarray,
        domain: str = "generic",
    ) -> Specialist:
        def process_fn(x: np.ndarray) -> Tuple[np.ndarray, float]:
            if x.shape[0] != weight_matrix.shape[1]:
                x_padded = np.zeros(weight_matrix.shape[1])
                x_padded[:min(len(x), weight_matrix.shape[1])] = x[:weight_matrix.shape[1]]
                x = x_padded
            output = np.tanh(weight_matrix @ x)
            salience = float(np.mean(np.abs(output)))
            return output, salience

        sp = Specialist(specialist_id, process_fn, domain)
        self.register_specialist(sp)
        return sp

    def phi_gwt(self) -> float:
        if len(self._workspace) < 2:
            return 0.0
        contents = [m.content for m in self._workspace[-4:]]
        if len(contents) < 2:
            return 0.0
        from itertools import combinations
        mutual_info_sum = 0.0
        for a, b in combinations(contents, 2):
            min_len = min(len(a), len(b))
            corr = float(np.corrcoef(a[:min_len], b[:min_len])[0, 1])
            if np.isfinite(corr):
                mutual_info_sum += abs(corr)
        n_pairs = max(1, len(contents) * (len(contents) - 1) // 2)
        return mutual_info_sum / n_pairs