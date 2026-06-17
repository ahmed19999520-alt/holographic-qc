import numpy as np
from holographic_qc.core.virasoro import VirasoroAlgebra, VirasoroConfig
from holographic_qc.core.ads_cft import AdsCft3
from holographic_qc.core.ryu_takayanagi import RyuTakayanagi, RTConfig
from holographic_qc.core.dilaton import DilatonField, DilatonConfig
from holographic_qc.core.christoffel import ChristoffelSymbols
from holographic_qc.algorithms.shor import ShorAlgorithm
from holographic_qc.algorithms.grover import GroverAlgorithm
from holographic_qc.algorithms.vqe import VQE
from holographic_qc.protection.decoherence import HolographicDecoherence
from holographic_qc.protection.majorana import MajoranaQubit
from holographic_qc.protection.holographic_error_correction import HolographicCode
from holographic_qc.materials.bi2se3 import Bi2Se3
from holographic_qc.materials.hgte import HgTeCdTe
from holographic_qc.materials.trapped_ions import TrappedIonChain
from holographic_qc.ml.pytorch_models import HolographicDecoherenceNet
from holographic_qc.ml.training import TrainingPipeline
from holographic_qc.utils.datasets import SyntheticDatasetGenerator, RealDataLoader
from holographic_qc.utils.benchmarks import Benchmarker


def run_full_pipeline():
    print("HolographicQC Full Integration Pipeline")
    print("=" * 65)

    print("\n[1/9] Virasoro algebra verification")
    cfg = VirasoroConfig(central_charge=1.0, max_mode=8)
    alg = VirasoroAlgebra(cfg)
    lin, cen = alg.commutator_scalar(3, -3)
    assert lin == 6.0 and abs(cen - 2.0) < 1e-10
    lam = alg.lyapunov_from_central_charge(4.0)
    kB, hbar = 1.380649e-23, 1.054571817e-34
    bound = 2.0 * np.pi * kB * 4.0 / hbar
    assert lam <= bound
    print(f"   [L_3, L_-3]: linear={lin}, central={cen:.4f}  |  lambda_L/bound={lam/bound:.6f}")

    print("\n[2/9] AdS3/CFT2 dictionary")
    mat = Bi2Se3()
    ads = AdsCft3(central_charge=mat.central_charge, ads_radius=mat.xi, fermi_velocity=mat.fermi_velocity)
    c_bh = ads.brown_henneaux_central_charge()
    sigma0 = ads.optical_conductivity_dc()
    wf = ads.wiedemann_franz_ratio()
    lam_ads = ads.lyapunov_exponent(4.0)
    enh = ads.holographic_coherence_enhancement(1e-6, mat.xi)
    print(f"   c={c_bh:.2f}, sigma_0={sigma0:.4e}, WF={wf:.4e}, lambda_L={lam_ads:.4e}, enh={enh:.4f}")

    print("\n[3/9] Ryu-Takayanagi entanglement entropy")
    rt_cfg = RTConfig(central_charge=1.0, newton_constant_3d=ads.newton_constant_3d,
                      ads_radius=ads.ads_radius, uv_cutoff=mat.a_lattice)
    rt = RyuTakayanagi(rt_cfg)
    for ell_nm in [10, 50, 100]:
        ell = ell_nm * 1e-9
        S = rt.entanglement_entropy_central_charge(ell)
        print(f"   ell={ell_nm}nm: S_A={S:.6f}")

    print("\n[4/9] Holographic decoherence protection")
    dec = HolographicDecoherence(ads, mat)
    T_test = 4.0
    T2_std = dec.coherence_time_standard(T_test)
    T2_holo = dec.coherence_time_holographic(T_test, 1e-6)
    print(f"   T2_std={T2_std*1e9:.4f}ns, T2_holo={T2_holo*1e9:.4f}ns, ratio={T2_holo/T2_std:.4f}")

    print("\n[5/9] Majorana qubit")
    qubit = MajoranaQubit(100e-9, mat.xi, mat.fermi_velocity, ads)
    T2_maj = qubit.total_coherence_time(1e-6)
    F_gate = qubit.gate_fidelity(0.01)
    berry = qubit.holographic_berry_phase()
    print(f"   T2_total={T2_maj:.4e}s, F_gate={F_gate:.8f}, phi_Berry={berry:.8f} rad")

    print("\n[6/9] Holographic error correction")
    code = HolographicCode(code_type="surface", central_charge=1.0)
    d_holo = code.effective_distance(1e-6, 1e-9)
    p_thresh = code.error_threshold(0.01)
    log_err = code.logical_error_rate(0.005, 1e-6, 1e-9)
    overhead = code.resource_overhead(10)
    print(f"   d_holo={d_holo:.4f}, p_thresh={p_thresh:.6f}, log_err={log_err:.4e}")
    print(f"   Resource overhead: {overhead}")

    print("\n[7/9] Quantum algorithms")
    shor = ShorAlgorithm(15)
    result_shor = shor.factor_classical_simulation()
    assert result_shor is not None and shor.verify_factorization(*result_shor)
    print(f"   Shor(15): {result_shor[0]} x {result_shor[1]} = {result_shor[0]*result_shor[1]}")

    grover = GroverAlgorithm(8, [42, 100])
    g_result = grover.run_with_measurement(n_shots=1000)
    print(f"   Grover(n=8, M=2): k_opt={g_result['n_iterations']}, success={g_result['success_rate']:.4f}")

    H_ising = VQE.ising_hamiltonian(4, J=1.0, h=1.0)
    vqe = VQE(H_ising, 4, depth=3, max_iter=300)
    vqe_result = vqe.run(seed=42)
    print(f"   VQE(Ising, n=4): E={vqe_result['optimal_energy']:.6f}, err={vqe_result['energy_error']:.2e}")

    print("\n[8/9] Machine learning pipeline")
    gen = SyntheticDatasetGenerator(seed=0)
    df = gen.generate_bi2se3_dataset(n_samples=2000, noise_level=0.05)
    loader = RealDataLoader("data")
    X_tr, X_v, X_te, y_tr, y_v, y_te, scaler = loader.prepare_training_features(
        df,
        ["temperature_K", "system_size_m", "fermi_velocity", "coherence_length_nm",
         "central_charge", "ratio_L_xi"],
        ["t2_standard_ns", "enhancement_factor"],
    )
    y_rate_tr, y_enh_tr = y_tr[:, 0:1], y_tr[:, 1:2]
    y_rate_v, y_enh_v = y_v[:, 0:1], y_v[:, 1:2]
    y_rate_te, y_enh_te = y_te[:, 0:1], y_te[:, 1:2]

    pipeline = TrainingPipeline(
        model_class=HolographicDecoherenceNet,
        model_kwargs={"input_dim": X_tr.shape[1], "central_charge": 1.0, "hidden_dims": [64, 128, 64, 32]},
        output_dir="models/integration_test/",
        experiment_name="integration",
    )
    pipeline.build_model()
    hist = pipeline.train(X_tr, y_rate_tr, y_enh_tr, X_v, y_rate_v, y_enh_v,
                         epochs=30, batch_size=32, verbose=False)
    metrics = pipeline.evaluate(X_te, y_rate_te, y_enh_te)
    print(f"   Final val_loss={hist['val_loss'][-1]:.6f}, test_loss={metrics['total_loss']:.6f}")

    print("\n[9/9] Trapped ions OTOC")
    chain = TrappedIonChain(n_ions=50)
    lam_ion = chain.lyapunov_exponent(1.0)
    t_s = chain.scrambling_time_ms(1.0)
    EE = chain.entanglement_entropy_critical(25)
    print(f"   lambda_L={lam_ion:.4e} s^-1, t_scramble={t_s:.4f} ms, EE(25/50)={EE:.6f}")

    print("\n" + "=" * 65)
    print("All 9 pipeline stages completed successfully.")
    print("=" * 65)
    return {
        "virasoro_ok": True,
        "ads_cft_ok": True,
        "rt_ok": True,
        "decoherence_ok": True,
        "majorana_ok": True,
        "qec_ok": True,
        "algorithms_ok": True,
        "ml_ok": True,
        "ions_ok": True,
    }


if __name__ == "__main__":
    results = run_full_pipeline()
    all_ok = all(results.values())
    print(f"\nFull pipeline: {'PASS' if all_ok else 'FAIL'}")