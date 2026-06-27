from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class NetworkConfig:
    n_input: int = 64
    n_hidden: int = 128
    n_output: int = 32
    dt_ms: float = 1.0
    tau_membrane_ms: float = 20.0
    tau_synapse_ms: float = 5.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    v_rest: float = 0.0
    refractory_period_ms: float = 2.0
    stdp_tau_plus: float = 20.0
    stdp_tau_minus: float = 20.0
    stdp_A_plus: float = 0.01
    stdp_A_minus: float = 0.0105
    max_weight: float = 1.0
    min_weight: float = 0.0


class LIFNeuron:
    def __init__(self, config: NetworkConfig, neuron_id: int = 0):
        self.id = neuron_id
        self.cfg = config
        self.v = config.v_rest
        self.spike = False
        self._refractory_remaining = 0.0
        self._spike_times: List[float] = []
        self._t = 0.0
        self._n_spikes = 0
        self._last_spike_time = -np.inf

    def step(self, I_ext: float) -> bool:
        self._t += self.cfg.dt_ms
        if self._refractory_remaining > 0:
            self._refractory_remaining -= self.cfg.dt_ms
            self.v = self.cfg.v_reset
            self.spike = False
            return False
        tau = self.cfg.tau_membrane_ms
        dt = self.cfg.dt_ms
        dv = dt / tau * (-(self.v - self.cfg.v_rest) + I_ext)
        self.v += dv
        if self.v >= self.cfg.v_threshold:
            self.spike = True
            self._spike_times.append(self._t)
            self._last_spike_time = self._t
            self._n_spikes += 1
            self._refractory_remaining = self.cfg.refractory_period_ms
            self.v = self.cfg.v_reset
        else:
            self.spike = False
        return self.spike

    def firing_rate_hz(self, window_ms: float = 100.0) -> float:
        recent = sum(1 for t in self._spike_times if self._t - t <= window_ms)
        return recent / (window_ms * 1e-3)

    def reset(self):
        self.v = self.cfg.v_rest
        self.spike = False
        self._refractory_remaining = 0.0
        self._spike_times = []
        self._t = 0.0
        self._n_spikes = 0
        self._last_spike_time = -np.inf


class SpikingLayer:
    def __init__(self, n_neurons: int, config: NetworkConfig, layer_id: str = "layer"):
        self.n = n_neurons
        self.layer_id = layer_id
        self.neurons = [LIFNeuron(config, i) for i in range(n_neurons)]
        self._t = 0.0
        self.cfg = config

    def step(self, currents: np.ndarray) -> np.ndarray:
        assert len(currents) == self.n
        self._t += self.cfg.dt_ms
        spikes = np.array([n.step(I) for n, I in zip(self.neurons, currents)], dtype=np.float32)
        return spikes

    def firing_rates(self, window_ms: float = 100.0) -> np.ndarray:
        return np.array([n.firing_rate_hz(window_ms) for n in self.neurons])

    def membrane_potentials(self) -> np.ndarray:
        return np.array([n.v for n in self.neurons])

    def reset(self):
        for n in self.neurons:
            n.reset()
        self._t = 0.0


class STDPLearning:
    def __init__(self, config: NetworkConfig):
        self.cfg = config
        self._pre_trace: np.ndarray = None
        self._post_trace: np.ndarray = None
        self._t = 0.0

    def initialize(self, n_pre: int, n_post: int):
        self._pre_trace = np.zeros(n_pre)
        self._post_trace = np.zeros(n_post)

    def update(
        self,
        W: np.ndarray,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
    ) -> np.ndarray:
        if self._pre_trace is None:
            self.initialize(len(pre_spikes), len(post_spikes))
        dt = self.cfg.dt_ms
        tau_plus = self.cfg.stdp_tau_plus
        tau_minus = self.cfg.stdp_tau_minus
        self._pre_trace *= np.exp(-dt / tau_plus)
        self._post_trace *= np.exp(-dt / tau_minus)
        self._pre_trace += pre_spikes
        self._post_trace += post_spikes
        A_plus = self.cfg.stdp_A_plus
        A_minus = self.cfg.stdp_A_minus
        dW = (
            A_plus * np.outer(post_spikes, self._pre_trace)
            - A_minus * np.outer(self._post_trace, pre_spikes)
        )
        W = np.clip(W + dW, self.cfg.min_weight, self.cfg.max_weight)
        self._t += dt
        return W

    def reset(self):
        if self._pre_trace is not None:
            self._pre_trace[:] = 0.0
        if self._post_trace is not None:
            self._post_trace[:] = 0.0
        self._t = 0.0


class SynapticTrace:
    def __init__(self, tau_ms: float, dt_ms: float):
        self.tau = tau_ms
        self.dt = dt_ms
        self._decay = np.exp(-dt_ms / tau_ms)
        self._value = 0.0

    def step(self, spike: float) -> float:
        self._value = self._decay * self._value + spike
        return self._value

    def reset(self):
        self._value = 0.0


class SpikingNetwork:
    def __init__(self, config: NetworkConfig):
        self.cfg = config
        self.input_layer = SpikingLayer(config.n_input, config, "input")
        self.hidden_layer = SpikingLayer(config.n_hidden, config, "hidden")
        self.output_layer = SpikingLayer(config.n_output, config, "output")
        rng = np.random.default_rng(42)
        self.W_ih = rng.uniform(0.0, 0.3, (config.n_hidden, config.n_input))
        self.W_ho = rng.uniform(0.0, 0.3, (config.n_output, config.n_hidden))
        self.W_hh = rng.uniform(0.0, 0.1, (config.n_hidden, config.n_hidden))
        np.fill_diagonal(self.W_hh, 0.0)
        self.stdp = STDPLearning(config)
        self._input_traces = [SynapticTrace(config.tau_synapse_ms, config.dt_ms)
                               for _ in range(config.n_input)]
        self._hidden_traces = [SynapticTrace(config.tau_synapse_ms, config.dt_ms)
                                for _ in range(config.n_hidden)]
        self._t = 0.0
        self._output_history: List[np.ndarray] = []
        self._n_steps = 0

    def step(self, input_rate: np.ndarray, learn: bool = True) -> np.ndarray:
        self._t += self.cfg.dt_ms
        self._n_steps += 1
        input_spikes = (np.random.rand(self.cfg.n_input) < (input_rate * self.cfg.dt_ms / 1000.0)).astype(float)
        in_currents = np.zeros(self.cfg.n_input)
        inp_spikes_out = self.input_layer.step(in_currents)
        for i, (trace, sp) in enumerate(zip(self._input_traces, input_spikes)):
            inp_spikes_out[i] = max(inp_spikes_out[i], sp)
            trace.step(inp_spikes_out[i])
        I_hidden = self.W_ih @ inp_spikes_out
        hidden_spikes = self.hidden_layer.step(I_hidden)
        I_recurrent = self.W_hh @ hidden_spikes
        hidden_spikes_final = self.hidden_layer.step(I_recurrent * 0.1)
        I_output = self.W_ho @ hidden_spikes_final
        output_spikes = self.output_layer.step(I_output)
        if learn:
            self.W_ih = self.stdp.update(self.W_ih, inp_spikes_out, hidden_spikes_final)
        self._output_history.append(output_spikes.copy())
        if len(self._output_history) > 1000:
            self._output_history.pop(0)
        return output_spikes

    def run(
        self, input_rates: np.ndarray, n_steps: int = 100, learn: bool = True
    ) -> np.ndarray:
        spike_counts = np.zeros((n_steps, self.cfg.n_output))
        for t in range(n_steps):
            spikes = self.step(input_rates, learn=learn)
            spike_counts[t] = spikes
        return spike_counts

    def output_firing_rates(self, window_ms: float = 100.0) -> np.ndarray:
        return self.output_layer.firing_rates(window_ms)

    def population_activity(self) -> dict:
        return {
            "input_rates": self.input_layer.firing_rates().tolist(),
            "hidden_rates": self.hidden_layer.firing_rates().tolist(),
            "output_rates": self.output_layer.firing_rates().tolist(),
            "W_ih_mean": float(np.mean(self.W_ih)),
            "W_ho_mean": float(np.mean(self.W_ho)),
            "n_steps": self._n_steps,
        }

    def reset(self):
        self.input_layer.reset()
        self.hidden_layer.reset()
        self.output_layer.reset()
        self.stdp.reset()
        for tr in self._input_traces + self._hidden_traces:
            tr.reset()
        self._t = 0.0
        self._n_steps = 0
        self._output_history = []

    def encode_rate(self, signal: np.ndarray, max_rate_hz: float = 100.0) -> np.ndarray:
        n = min(len(signal), self.cfg.n_input)
        rates = np.zeros(self.cfg.n_input)
        rates[:n] = np.abs(signal[:n]) / (np.max(np.abs(signal[:n])) + 1e-10) * max_rate_hz
        return rates

    def decode_rate(self, window_ms: float = 100.0) -> np.ndarray:
        return self.output_layer.firing_rates(window_ms)