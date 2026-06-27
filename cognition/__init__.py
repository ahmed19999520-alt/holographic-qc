from holographic_qc.cognition.global_workspace import (
    GlobalWorkspace, WorkspaceConfig, Specialist, BroadcastMessage
)
from holographic_qc.cognition.integrated_information import (
    IntegratedInformationTheory, IITConfig, CauseEffectStructure
)
from holographic_qc.cognition.vector_memory import (
    VectorMemory, EpisodicMemory, SemanticMemory, MemoryConfig
)
from holographic_qc.cognition.goal_system import (
    GoalSystem, Goal, GoalState, MotivationEngine
)
from holographic_qc.cognition.self_model import (
    SelfModel, SelfModelConfig, MetacognitiveMonitor
)
from holographic_qc.cognition.spiking_networks import (
    SpikingNetwork, LIFNeuron, SpikingLayer, STDPLearning, NetworkConfig
)
from holographic_qc.cognition.continual_learning import (
    ContinualLearner, EWC, ProgressiveNet, ExperienceReplay
)
from holographic_qc.cognition.symbolic_reasoning import (
    SymbolicReasoner, KnowledgeBase, Fact, Rule, InferenceEngine
)
from holographic_qc.cognition.cognitive_architecture import (
    CognitiveArchitecture, ArchitectureConfig, CognitiveCycle
)

__version_cognition__ = "0.1.0"

__all__ = [
    "GlobalWorkspace", "WorkspaceConfig", "Specialist", "BroadcastMessage",
    "IntegratedInformationTheory", "IITConfig", "CauseEffectStructure",
    "VectorMemory", "EpisodicMemory", "SemanticMemory", "MemoryConfig",
    "GoalSystem", "Goal", "GoalState", "MotivationEngine",
    "SelfModel", "SelfModelConfig", "MetacognitiveMonitor",
    "SpikingNetwork", "LIFNeuron", "SpikingLayer", "STDPLearning", "NetworkConfig",
    "ContinualLearner", "EWC", "ProgressiveNet", "ExperienceReplay",
    "SymbolicReasoner", "KnowledgeBase", "Fact", "Rule", "InferenceEngine",
    "CognitiveArchitecture", "ArchitectureConfig", "CognitiveCycle",
]