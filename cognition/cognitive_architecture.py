from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from holographic_qc.cognition.global_workspace import GlobalWorkspace, WorkspaceConfig
from holographic_qc.cognition.integrated_information import IntegratedInformationTheory, IITConfig
from holographic_qc.cognition.vector_memory import VectorMemory, MemoryConfig
from holographic_qc.cognition.goal_system import GoalSystem, MotivationEngine
from holographic_qc.cognition.self_model import SelfModel, SelfModelConfig
from holographic_qc.cognition.spiking_networks import SpikingNetwork, NetworkConfig
from holographic_qc.cognition.continual_learning import ContinualLearner, ContinualConfig, ExperienceReplay, ReplayBuffer
from holographic_qc.cognition.symbolic_reasoning import SymbolicReasoner
from holographic_qc.core.virasoro import VirasoroAlgebra, VirasoroConfig
from holographic_qc.core.ads_cft import AdsCft3


@dataclass
class ArchitectureConfig:
    workspace: WorkspaceConfig = field(default_factory=lambda: WorkspaceConfig(
        capacity=7, broadcast_threshold=0.5, workspace_dim=256, n_specialists=6
    ))
    iit: IITConfig = field(default_factory=lambda: IITConfig(
        n_elements=8, connectivity_density=0.5, phi_approximation="lz"
    ))
    memory: MemoryConfig = field(default_factory=lambda: MemoryConfig(
        embedding_dim=256, episodic_capacity=5000, semantic_capacity=2000
    ))
    self_model: SelfModelConfig = field(default_factory=lambda: SelfModelConfig(
        state_dim=256, history_len=100
    ))
    spiking: NetworkConfig = field(default_factory=lambda: NetworkConfig(
        n_input=64, n_hidden=128, n_output=32
    ))
    continual: ContinualConfig = field(default_factory=lambda: ContinualConfig(
        ewc_lambda=500.0, n_epochs_per_task=5
    ))
    virasoro_c: float = 1.0
    ads_radius: float = 1.1e-9
    cycle_dt_ms: float = 10.0
    n_homeostatic_vars: int = 4


@dataclass
class CognitiveCycle:
    cycle_id: int
    timestamp: float
    inputs: Dict[str, Any]
    percept: Optional[Any]
    memory_retrievals: List[Any]
    goal_id: Optional[str]
    phi: float
    gwt_ignition: float
    self_state_norm: float
    symbolic_inferences: int
    spike_rates: np.ndarray
    outputs: Dict[str, Any]
    duration_ms: float


class PerceptionEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionDecoder(nn.Module):
    def __init__(self, embed_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CognitiveArchitecture:
    def __init__(self, config: ArchitectureConfig, input_dim: int = 64, action_dim: int = 16):
        self.cfg = config
        self.input_dim = input_dim
        self.action_dim = action_dim

        self.workspace = GlobalWorkspace(config.workspace)
        self.iit = IntegratedInformationTheory(config.iit)
        self.iit.random_connectivity(seed=42)
        self.memory = VectorMemory(config.memory)
        homeostatic_vars = [f"need_{i}" for i in range(config.n_homeostatic_vars)]
        set_points = {v: 0.7 for v in homeostatic_vars}
        self.motivation = MotivationEngine(homeostatic_vars, set_points)
        self.goals = GoalSystem(self.motivation)
        self.self_model = SelfModel(config.self_model)
        self.snn = SpikingNetwork(config.spiking)
        self.symbolic = SymbolicReasoner()

        self.perception_encoder = PerceptionEncoder(input_dim, config.memory.embedding_dim)
        self.action_decoder = ActionDecoder(config.workspace.workspace_dim, action_dim)

        self.virasoro = VirasoroAlgebra(VirasoroConfig(central_charge=config.virasoro_c, max_mode=6))
        self.ads = AdsCft3(central_charge=config.virasoro_c, ads_radius=config.ads_radius)

        self._cycle_history: List[CognitiveCycle] = []
        self._n_cycles: int = 0
        self._continual_model = nn.Sequential(
            nn.Linear(config.memory.embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self._continual_learner = ContinualLearner(
            self._continual_model, config.continual, strategy="ewc"
        )

        self._register_default_specialists()
        self._register_default_goals()
        self._register_default_symbolic_knowledge()

    def _register_default_specialists(self):
        embed_dim = self.cfg.memory.embedding_dim
        n_sp = self.cfg.workspace.n_specialists
        for i in range(n_sp):
            W = np.random.randn(64, embed_dim) * 0.1
            self.workspace.create_specialist(
                f"specialist_{i}",
                W,
                domain=["perception", "memory", "reasoning", "action", "emotion", "attention"][i % 6],
            )

    def _register_default_goals(self):
        self.goals.add_goal("maintain_homeostasis", utility=0.9, priority=1.0)
        self.goals.add_goal("encode_new_information", utility=0.7, priority=0.8)
        self.goals.add_goal("retrieve_relevant_memory", utility=0.6, priority=0.7)
        self.goals.add_goal("resolve_symbolic_inconsistency", utility=0.5, priority=0.6)

    def _register_default_symbolic_knowledge(self):
        self.symbolic.assert_fact("is_a", "holographic_qubit", "quantum_system")
        self.symbolic.assert_fact("is_a", "virasoro_algebra", "symmetry_algebra")
        self.symbolic.assert_fact("implements", "ads_cft", "holographic_protection")
        self.symbolic.assert_fact("is_a", "majorana_fermion", "quasiparticle")
        self.symbolic.add_rule(
            "entanglement_implies_protection",
            head=("protected", ("?X",)),
            body=[("is_a", ("?X", "quantum_system")), ("implements", ("ads_cft", "holographic_protection"))],
            confidence=0.85,
        )
        self.symbolic.add_rule(
            "consciousness_from_phi",
            head=("conscious", ("?X",)),
            body=[("is_a", ("?X", "cognitive_system")), ("has_phi_above_threshold", ("?X",))],
            confidence=0.7,
        )

    def _encode_perception(self, raw_input: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.FloatTensor(raw_input).unsqueeze(0)
            if x_t.shape[-1] != self.input_dim:
                padded = torch.zeros(1, self.input_dim)
                n = min(x_t.shape[-1], self.input_dim)
                padded[0, :n] = x_t[0, :n]
                x_t = padded
            embedding = self.perception_encoder(x_t).squeeze(0).numpy()
        return embedding

    def _decode_action(self, global_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            g_t = torch.FloatTensor(global_state).unsqueeze(0)
            if g_t.shape[-1] != self.cfg.workspace.workspace_dim:
                padded = torch.zeros(1, self.cfg.workspace.workspace_dim)
                n = min(g_t.shape[-1], self.cfg.workspace.workspace_dim)
                padded[0, :n] = g_t[0, :n]
                g_t = padded
            action = self.action_decoder(g_t).squeeze(0).numpy()
        return action

    def cycle(self, raw_input: np.ndarray, context: Dict[str, Any] = None) -> CognitiveCycle:
        t0 = time.perf_counter()
        self._n_cycles += 1
        context = context or {}

        embedding = self._encode_perception(raw_input)

        snn_rates = self.snn.encode_rate(raw_input[:self.cfg.spiking.n_input])
        spike_output = self.snn.step(snn_rates, learn=True)
        snn_output_rates = self.snn.output_firing_rates(window_ms=50.0)

        memory_results = self.memory.retrieve(embedding, memory_type="episodic", k=3)

        specialist_stimuli = {}
        for i, sp_id in enumerate(list(self.workspace.specialists.keys())[:4]):
            sp_stimulus = embedding + 0.1 * np.random.randn(len(embedding))
            specialist_stimuli[sp_id] = sp_stimulus[:len(embedding)]

        winner = self.workspace.step(specialist_stimuli)
        global_state = self.workspace.global_state()
        gwt_ignition = self.workspace.ignition_score()

        self.self_model.observe_state(global_state[:self.cfg.self_model.state_dim])

        phi = self.iit.compute_phi(state=self._n_cycles % self.iit.cfg.n_states)

        goal_id = self.goals.cycle()

        holographic_enh = self.ads.holographic_coherence_enhancement(1e-6, self.cfg.ads_radius)
        lam = self.virasoro.lyapunov_from_central_charge(4.0)
        new_facts = self.symbolic.infer()

        importance = float(gwt_ignition * (1.0 + phi * 0.1))
        self.memory.store(
            content={"input_summary": raw_input[:4].tolist(), "cycle": self._n_cycles},
            embedding=embedding,
            memory_type="episodic",
            importance=importance,
        )

        for hv in self.motivation.vars:
            current = self.motivation._current_values[hv]
            delta = 0.01 * (np.random.randn() * 0.1)
            self.motivation.update(hv, current + delta)

        if phi > 0.3:
            self.symbolic.assert_fact("has_phi_above_threshold", "cognitive_system", confidence=float(phi))
            self.symbolic.assert_fact("is_a", "cognitive_system", "cognitive_system")
            self.symbolic.infer()

        action = self._decode_action(global_state)

        t1 = time.perf_counter()
        cycle_result = CognitiveCycle(
            cycle_id=self._n_cycles,
            timestamp=t0,
            inputs={"raw": raw_input[:8].tolist(), "context": context},
            percept=winner.content.tolist() if winner is not None else None,
            memory_retrievals=[(str(c)[:50], sim) for c, sim in memory_results],
            goal_id=goal_id,
            phi=phi,
            gwt_ignition=gwt_ignition,
            self_state_norm=float(np.linalg.norm(self.self_model.state_vector())),
            symbolic_inferences=len(new_facts),
            spike_rates=snn_output_rates,
            outputs={
                "action": action.tolist(),
                "holographic_enh": holographic_enh,
                "lyapunov": lam,
                "most_urgent_need": self.motivation.most_urgent_need(),
            },
            duration_ms=(t1 - t0) * 1000.0,
        )
        self._cycle_history.append(cycle_result)
        if len(self._cycle_history) > 500:
            self._cycle_history.pop(0)
        return cycle_result

    def run(self, inputs: List[np.ndarray], verbose: bool = True) -> List[CognitiveCycle]:
        results = []
        for i, inp in enumerate(inputs):
            cycle = self.cycle(inp)
            results.append(cycle)
            if verbose and i % 10 == 0:
                print(
                    f"Cycle {cycle.cycle_id:04d} | "
                    f"phi={cycle.phi:.4f} | "
                    f"ignition={cycle.gwt_ignition:.4f} | "
                    f"self_norm={cycle.self_state_norm:.4f} | "
                    f"goal={cycle.goal_id} | "
                    f"dt={cycle.duration_ms:.2f}ms"
                )
        return results

    def global_report(self) -> dict:
        if not self._cycle_history:
            return {}
        phi_vals = [c.phi for c in self._cycle_history]
        ignition_vals = [c.gwt_ignition for c in self._cycle_history]
        dt_vals = [c.duration_ms for c in self._cycle_history]
        return {
            "n_cycles": self._n_cycles,
            "mean_phi": float(np.mean(phi_vals)),
            "max_phi": float(np.max(phi_vals)),
            "mean_gwt_ignition": float(np.mean(ignition_vals)),
            "mean_cycle_ms": float(np.mean(dt_vals)),
            "memory_stats": self.memory.statistics(),
            "goal_stats": self.goals.statistics(),
            "self_model": self.self_model.narrative_self_description(),
            "symbolic_stats": self.symbolic.statistics(),
            "workspace_report": self.workspace.conscious_access_report(),
            "snn_activity": self.workspace.specialists.__len__(),
            "holographic_enh": self.ads.holographic_coherence_enhancement(1e-6, self.cfg.ads_radius),
            "virasoro_lyapunov_4K": self.virasoro.lyapunov_from_central_charge(4.0),
        }