import numpy as np
import pytest
from holographic_qc.materials.bi2se3 import Bi2Se3
from holographic_qc.materials.hgte import HgTeCdTe
from holographic_qc.materials.trapped_ions import TrappedIonChain
from holographic_qc.core.ads_cft import AdsCft3
from holographic_qc.core.virasoro import VirasoroAlgebra, VirasoroConfig
from holographic_qc.core.ryu_takayanagi import RyuTakayanagi, RTConfig
from holographic_qc.protection.decoherence import HolographicDecoherence
from holographic_qc.utils.datasets import SyntheticDatasetGenerator


@pytest.fixture(scope="session")
def bi2se3():
    return Bi2Se3()


@pytest.fixture(scope="session")
def hgte():
    return HgTeCdTe()


@pytest.fixture(scope="session")
def ion_chain_small():
    return TrappedIonChain(n_ions=8, transverse_field_ratio=1.0)


@pytest.fixture(scope="session")
def ads_c1(bi2se3):
    return AdsCft3(
        central_charge=bi2se3.central_charge,
        ads_radius=bi2se3.xi,
        fermi_velocity=bi2se3.fermi_velocity,
    )


@pytest.fixture(scope="session")
def ads_c2(hgte):
    return AdsCft3(
        central_charge=hgte.central_charge,
        ads_radius=hgte.xi,
        fermi_velocity=hgte.fermi_velocity,
    )


@pytest.fixture(scope="session")
def virasoro_c1():
    return VirasoroAlgebra(VirasoroConfig(central_charge=1.0, max_mode=8))


@pytest.fixture(scope="session")
def virasoro_c2():
    return VirasoroAlgebra(VirasoroConfig(central_charge=2.0, max_mode=8))


@pytest.fixture(scope="session")
def rt_c1(bi2se3, ads_c1):
    return RyuTakayanagi(RTConfig(
        central_charge=bi2se3.central_charge,
        newton_constant_3d=ads_c1.newton_constant_3d,
        ads_radius=ads_c1.ads_radius,
        uv_cutoff=bi2se3.a_lattice,
    ))


@pytest.fixture(scope="session")
def holo_dec_c1(ads_c1, bi2se3):
    return HolographicDecoherence(ads_system=ads_c1, material=bi2se3)


@pytest.fixture(scope="session")
def holo_dec_c2(ads_c2, hgte):
    return HolographicDecoherence(ads_system=ads_c2, material=hgte)


@pytest.fixture(scope="session")
def small_dataset():
    gen = SyntheticDatasetGenerator(seed=99)
    return gen.generate_bi2se3_dataset(n_samples=200, noise_level=0.03)


@pytest.fixture(scope="session")
def ee_dataset():
    gen = SyntheticDatasetGenerator(seed=99)
    return gen.generate_entanglement_entropy_dataset(
        n_samples=400, central_charges=[0.5, 1.0, 2.0]
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with -m 'not slow')")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "tf: marks tests requiring TensorFlow")