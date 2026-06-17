from holographic_qc.ml.pytorch_models import (
    HolographicDecoherenceNet as TorchDecoherenceNet,
    EntanglementEntropyNet as TorchEENet,
    OTOCNet as TorchOTOCNet,
    HolographicTrainer as TorchTrainer,
    VirasoroEquivariantLayer as TorchVirasoroLayer,
)

try:
    from holographic_qc.ml.tensorflow_models import (
        HolographicDecoherenceNet as TFDecoherenceNet,
        EntanglementEntropyNet as TFEENet,
        OTOCNet as TFOTOCNet,
        HolographicTrainer as TFTrainer,
        build_decoherence_model,
    )
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

from holographic_qc.ml.training import TrainingPipeline, CrossValidationRunner

__all__ = [
    "TorchDecoherenceNet",
    "TorchEENet",
    "TorchOTOCNet",
    "TorchTrainer",
    "TorchVirasoroLayer",
    "TrainingPipeline",
    "CrossValidationRunner",
]

if _TF_AVAILABLE:
    __all__ += [
        "TFDecoherenceNet",
        "TFEENet",
        "TFOTOCNet",
        "TFTrainer",
        "build_decoherence_model",
    ]