from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import deque


@dataclass
class SelfModelConfig:
    state_dim: int = 128
    history_len: int = 100
    prediction_horizon: int = 10
    metacognitive_threshold: float = 0.3
    self_update_rate: float = 0.05


class StatePredictor:
    def __init__(self, state_dim: int, horizon: int):
        self.state_dim = state_dim
        self.horizon = horizon
        self._W = np.random.randn(state_dim, state_dim) * 0.01
        self._b = np.zeros(state_dim)
        self._lr = 1e-3
        self._n_updates = 0

    def predict(self, state: np.ndarray, n_steps: int = 1) -> List[np.ndarray]:
        predictions = []
        current = state.copy()
        for _ in range(n_steps):
            next_s = np.tanh(self._W @ current + self._b)
            predictions.append(next_s)
            current = next_s
        return predictions

    def update(self, state: np.ndarray, next_state: np.ndarray):
        predicted = np.tanh(self._W @ state + self._b)
        error = next_state - predicted
        grad_W = np.outer(error * (1 - predicted**2), state)
        grad_b = error * (1 - predicted**2)
        self._W += self._lr * grad_W
        self._b += self._lr * grad_b
        self._n_updates += 1
        return float(np.mean(error**2))


class MetacognitiveMonitor:
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self._confidence_history: deque = deque(maxlen=200)
        self._error_history: deque = deque(maxlen=200)
        self._intervention_count: int = 0

    def record_confidence(self, confidence: float, actual_correct: bool):
        error = abs(confidence - float(actual_correct))
        self._confidence_history.append(confidence)
        self._error_history.append(error)

    def calibration_error(self) -> float:
        if len(self._error_history) < 2:
            return 0.0
        return float(np.mean(self._error_history))

    def overconfidence_rate(self) -> float:
        if not self._confidence_history:
            return 0.0
        conf = np.array(self._confidence_history)
        err = np.array(self._error_history)
        over = np.sum((conf > 0.7) & (err > 0.3))
        return float(over / max(len(conf), 1))

    def should_intervene(self, confidence: float) -> bool:
        if confidence < self.threshold:
            self._intervention_count += 1
            return True
        calibration = self.calibration_error()
        if calibration > 0.2 and confidence > 0.8:
            self._intervention_count += 1
            return True
        return False

    def uncertainty_estimate(self, predictions: List[float]) -> float:
        if len(predictions) < 2:
            return 0.5
        return float(np.std(predictions))

    def statistics(self) -> dict:
        return {
            "n_records": len(self._confidence_history),
            "mean_confidence": float(np.mean(self._confidence_history)) if self._confidence_history else 0.0,
            "calibration_error": self.calibration_error(),
            "overconfidence_rate": self.overconfidence_rate(),
            "intervention_count": self._intervention_count,
        }


class SelfModel:
    def __init__(self, config: SelfModelConfig):
        self.cfg = config
        self._state: np.ndarray = np.zeros(config.state_dim)
        self._state_history: deque = deque(maxlen=config.history_len)
        self._predictor = StatePredictor(config.state_dim, config.prediction_horizon)
        self.metacognition = MetacognitiveMonitor(config.metacognitive_threshold)
        self._capabilities: Dict[str, float] = {}
        self._beliefs: Dict[str, float] = {}
        self._n_self_updates: int = 0
        self._prediction_errors: deque = deque(maxlen=100)
        self._identity_vector: np.ndarray = np.random.randn(config.state_dim)
        self._identity_vector /= np.linalg.norm(self._identity_vector)

    def observe_state(self, external_state: np.ndarray):
        if len(external_state) != self.cfg.state_dim:
            ext = np.zeros(self.cfg.state_dim)
            n = min(len(external_state), self.cfg.state_dim)
            ext[:n] = external_state[:n]
            external_state = ext
        alpha = self.cfg.self_update_rate
        prev_state = self._state.copy()
        self._state = (1 - alpha) * self._state + alpha * external_state
        self._state_history.append(self._state.copy())
        if len(self._state_history) > 1:
            pred_error = self._predictor.update(prev_state, self._state)
            self._prediction_errors.append(pred_error)
        self._n_self_updates += 1

    def predict_future_states(self, n_steps: int = None) -> List[np.ndarray]:
        n_steps = n_steps or self.cfg.prediction_horizon
        return self._predictor.predict(self._state, n_steps)

    def self_consistency(self) -> float:
        if len(self._state_history) < 3:
            return 1.0
        recent = list(self._state_history)[-10:]
        variances = np.var(np.stack(recent), axis=0)
        return float(1.0 - np.mean(variances))

    def register_capability(self, capability: str, proficiency: float):
        proficiency = float(np.clip(proficiency, 0.0, 1.0))
        if capability in self._capabilities:
            old = self._capabilities[capability]
            self._capabilities[capability] = 0.8 * old + 0.2 * proficiency
        else:
            self._capabilities[capability] = proficiency

    def update_belief(self, proposition: str, confidence: float):
        self._beliefs[proposition] = float(np.clip(confidence, 0.0, 1.0))

    def identity_similarity(self, other_state: np.ndarray) -> float:
        n = min(len(other_state), self.cfg.state_dim)
        other_norm = other_state[:n] / (np.linalg.norm(other_state[:n]) + 1e-10)
        self_slice = self._identity_vector[:n]
        return float(np.dot(self_slice, other_norm))

    def narrative_self_description(self) -> dict:
        pred_error_mean = float(np.mean(self._prediction_errors)) if self._prediction_errors else 0.0
        return {
            "state_norm": float(np.linalg.norm(self._state)),
            "self_consistency": self.self_consistency(),
            "n_capabilities": len(self._capabilities),
            "top_capabilities": sorted(self._capabilities.items(), key=lambda x: x[1], reverse=True)[:5],
            "n_beliefs": len(self._beliefs),
            "mean_belief_confidence": float(np.mean(list(self._beliefs.values()))) if self._beliefs else 0.0,
            "mean_prediction_error": pred_error_mean,
            "n_updates": self._n_self_updates,
            "metacognitive_calibration": self.metacognition.calibration_error(),
        }

    def counterfactual(self, hypothetical_input: np.ndarray, n_steps: int = 5) -> List[np.ndarray]:
        predictor_copy = StatePredictor(self.cfg.state_dim, self.cfg.prediction_horizon)
        predictor_copy._W = self._predictor._W.copy()
        predictor_copy._b = self._predictor._b.copy()
        if len(hypothetical_input) != self.cfg.state_dim:
            h = np.zeros(self.cfg.state_dim)
            n = min(len(hypothetical_input), self.cfg.state_dim)
            h[:n] = hypothetical_input[:n]
            hypothetical_input = h
        return predictor_copy.predict(hypothetical_input, n_steps)

    def state_vector(self) -> np.ndarray:
        return self._state.copy()