from holographic_qc.materials.bi2se3 import Bi2Se3
from holographic_qc.materials.hgte import HgTeCdTe
from holographic_qc.materials.trapped_ions import TrappedIonChain

MATERIALS_REGISTRY = {
    "Bi2Se3": Bi2Se3,
    "HgTeCdTe": HgTeCdTe,
    "TrappedIonChain": TrappedIonChain,
}

__all__ = ["Bi2Se3", "HgTeCdTe", "TrappedIonChain", "MATERIALS_REGISTRY"]