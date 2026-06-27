import numpy as np
from holographic_qc.cognition.cognitive_architecture import CognitiveArchitecture, ArchitectureConfig
from holographic_qc.cognition.global_workspace import WorkspaceConfig
from holographic_qc.cognition.integrated_information import IITConfig
from holographic_qc.cognition.vector_memory import MemoryConfig
from holographic_qc.cognition.self_model import SelfModelConfig
from holographic_qc.cognition.spiking_networks import NetworkConfig
from holographic_qc.cognition.continual_learning import ContinualConfig


def main():
    print("=" * 70)
    print("HolographicQC — Hybrid Cognitive Architecture Demo")
    print("=" * 70)

    cfg = ArchitectureConfig(
        workspace=WorkspaceConfig(capacity=7, broadcast_threshold=0.45,
                                  workspace_dim=128, n_specialists=6),
        iit=IITConfig(n_elements=6, connectivity_density=0.5, phi_approximation="lz"),
        memory=MemoryConfig(embedding_dim=128, episodic_capacity=2000),
        self_model=SelfModelConfig(state_dim=128, history_len=50),
        spiking=NetworkConfig(n_input=32, n_hidden=64, n_output=16),
        continual=ContinualConfig(ewc_lambda=200.0, n_epochs_per_task=3),
        virasoro_c=1.0,
        ads_radius=1.1e-9,
    )

    arch = CognitiveArchitecture(cfg, input_dim=32, action_dim=8)

    print("\n[Phase 1] Warmup — 20 random cycles")
    rng = np.random.default_rng(42)
    for i in range(20):
        inp = rng.randn(32).astype(np.float32)
        arch.cycle(inp, context={"phase": "warmup", "step": i})

    print("\n[Phase 2] Structured input — 30 cycles with patterns")
    for i in range(30):
        pattern = np.sin(np.linspace(0, 2 * np.pi * (i + 1) / 5, 32)).astype(np.float32)
        noise = rng.randn(32).astype(np.float32) * 0.1
        inp = pattern + noise
        cycle = arch.cycle(inp, context={"phase": "structured", "pattern": i % 5})
        if i % 5 == 0:
            print(
                f"  Cycle {cycle.cycle_id:03d}: "
                f"phi={cycle.phi:.4f}, "
                f"ignition={cycle.gwt_ignition:.4f}, "
                f"goal={cycle.goal_id}, "
                f"dt={cycle.duration_ms:.1f}ms"
            )

    print("\n[Phase 3] Symbolic reasoning test")
    arch.symbolic.assert_fact("is_a", "bi2se3_edge", "quantum_system")
    arch.symbolic.assert_fact("is_a", "bi2se3_edge", "cognitive_system")
    arch.symbolic.assert_fact("has_phi_above_threshold", "bi2se3_edge")
    arch.symbolic.add_rule(
        "topological_is_protected",
        head=("protected", ("?X",)),
        body=[("is_a", ("?X", "quantum_system"))],
        confidence=0.9,
    )
    new_facts = arch.symbolic.infer()
    print(f"  Derived {len(new_facts)} new facts via forward chaining:")
    for f in new_facts[:5]:
        print(f"    {f}")

    is_protected = arch.symbolic.prove("protected", "bi2se3_edge")
    print(f"  Bi2Se3 edge protected? {is_protected}")
    explanation = arch.symbolic.explain("protected", "bi2se3_edge")
    for line in explanation[:4]:
        print(f"    {line}")

    print("\n[Phase 4] Memory retrieval test")
    query = np.sin(np.linspace(0, 2 * np.pi, 128)).astype(np.float32)
    results = arch.memory.retrieve(query, memory_type="episodic", k=3)
    print(f"  Retrieved {len(results)} episodic memories (top-k=3):")
    for content, sim in results:
        print(f"    sim={sim:.4f}: {str(content)[:60]}")

    print("\n[Phase 5] Self model introspection")
    self_report = arch.self_model.narrative_self_description()
    print(f"  Self consistency: {self_report['self_consistency']:.6f}")
    print(f"  Mean prediction error: {self_report['mean_prediction_error']:.6e}")
    print(f"  N capabilities: {self_report['n_capabilities']}")
    print(f"  Metacognitive calibration: {self_report['metacognitive_calibration']:.4f}")

    print("\n[Phase 6] Spiking network activity")
    snn_activity = arch.snn.population_activity()
    hidden_rates = np.array(snn_activity["hidden_rates"])
    output_rates = np.array(snn_activity["output_rates"])
    print(f"  Mean hidden firing rate: {np.mean(hidden_rates):.4f} Hz")
    print(f"  Mean output firing rate: {np.mean(output_rates):.4f} Hz")
    print(f"  W_ih mean weight: {snn_activity['W_ih_mean']:.6f}")

    print("\n[Final Report]")
    report = arch.global_report()
    print(f"  Total cycles: {report['n_cycles']}")
    print(f"  Mean phi (IIT): {report['mean_phi']:.6f}")
    print(f"  Max phi: {report['max_phi']:.6f}")
    print(f"  Mean GWT ignition: {report['mean_gwt_ignition']:.6f}")
    print(f"  Mean cycle time: {report['mean_cycle_ms']:.2f} ms")
    print(f"  Memory (episodic): {report['memory_stats']['episodic']}")
    print(f"  Goals achieved: {report['goal_stats']['state_distribution']}")
    print(f"  Holographic enhancement: {report['holographic_enh']:.4f}")
    print(f"  Virasoro Lyapunov (4K): {report['virasoro_lyapunov_4K']:.4e} s^-1")
    print(f"  Symbolic KB: {report['symbolic_stats']['kb_stats']}")
    print("\nDemo complete.")


if __name__ == "__main__":
    main()