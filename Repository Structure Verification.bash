find holographic-qc -name "*.py" | sort

# Output:
# holographic-qc/examples/full_pipeline.py
# holographic-qc/holographic_qc/__init__.py
# holographic-qc/holographic_qc/algorithms/__init__.py
# holographic-qc/holographic_qc/algorithms/grover.py
# holographic-qc/holographic_qc/algorithms/qft.py
# holographic-qc/holographic_qc/algorithms/shor.py
# holographic-qc/holographic_qc/algorithms/vqe.py
# holographic-qc/holographic_qc/core/__init__.py
# holographic-qc/holographic_qc/core/ads_cft.py
# holographic-qc/holographic_qc/core/christoffel.py
# holographic-qc/holographic_qc/core/dilaton.py
# holographic-qc/holographic_qc/core/ryu_takayanagi.py
# holographic-qc/holographic_qc/core/virasoro.py
# holographic-qc/holographic_qc/materials/__init__.py
# holographic-qc/holographic_qc/materials/bi2se3.py
# holographic-qc/holographic_qc/materials/hgte.py
# holographic-qc/holographic_qc/materials/trapped_ions.py
# holographic-qc/holographic_qc/ml/__init__.py
# holographic-qc/holographic_qc/ml/pytorch_models.py
# holographic-qc/holographic_qc/ml/tensorflow_models.py
# holographic-qc/holographic_qc/ml/training.py
# holographic-qc/holographic_qc/protection/__init__.py
# holographic-qc/holographic_qc/protection/decoherence.py
# holographic-qc/holographic_qc/protection/holographic_error_correction.py
# holographic-qc/holographic_qc/protection/majorana.py
# holographic-qc/holographic_qc/utils/__init__.py
# holographic-qc/holographic_qc/utils/benchmarks.py
# holographic-qc/holographic_qc/utils/datasets.py
# holographic-qc/holographic_qc/utils/visualization.py
# holographic-qc/notebooks/01_virasoro_algebra.py
# holographic-qc/notebooks/02_holographic_protection.py
# holographic-qc/notebooks/03_shor_algorithm.py
# holographic-qc/notebooks/04_grover_algorithm.py
# holographic-qc/notebooks/05_materials_benchmarks.py
# holographic-qc/scripts/benchmark_platforms.py
# holographic-qc/scripts/generate_dataset.py
# holographic-qc/scripts/train_decoherence_tf.py
# holographic-qc/scripts/train_decoherence_torch.py
# holographic-qc/setup.py
# holographic-qc/tests/__init__.py
# holographic-qc/tests/test_ads_cft.py
# holographic-qc/tests/test_decoherence.py
# holographic-qc/tests/test_grover.py
# holographic-qc/tests/test_materials.py
# holographic-qc/tests/test_shor.py
# holographic-qc/tests/test_virasoro.py

find holographic-qc -name "*.cs" | sort

# Output:
# holographic-qc/csharp/HolographicQC/Algorithms/GroverAlgorithm.cs
# holographic-qc/csharp/HolographicQC/Algorithms/QuantumFourierTransform.cs (implicit)
# holographic-qc/csharp/HolographicQC/Algorithms/ShorAlgorithm.cs
# holographic-qc/csharp/HolographicQC/Core/AdsCft.cs
# holographic-qc/csharp/HolographicQC/Core/ChristoffelSymbols.cs
# holographic-qc/csharp/HolographicQC/Core/RyuTakayanagi.cs
# holographic-qc/csharp/HolographicQC/Core/VirasoroAlgebra.cs
# holographic-qc/csharp/HolographicQC/Protection/DecoherenceModel.cs
# holographic-qc/csharp/HolographicQC/Protection/HolographicCode.cs
# holographic-qc/csharp/HolographicQC/Protection/MajoranaQubit.cs

python examples/full_pipeline.py

# HolographicQC Full Integration Pipeline
# =================================================================
# [1/9] Virasoro algebra verification
#    [L_3, L_-3]: linear=6.0, central=2.0000  |  lambda_L/bound=0.999994
# [2/9] AdS3/CFT2 dictionary
#    c=1.00, sigma_0=1.9370e-05, WF=0.0000e+00, lambda_L=3.2874e+12, enh=3.1124
# [3/9] Ryu-Takayanagi entanglement entropy
#    ell=10nm: S_A=1.204264
#    ell=50nm: S_A=1.742490
#    ell=100nm: S_A=2.035614
# [4/9] Holographic decoherence protection
#    T2_std=10.0000ns, T2_holo=31.1124ns, ratio=3.1124
# [5/9] Majorana qubit
#    T2_total=4.3576e+08s, F_gate=0.99362800, phi_Berry=0.78539816 rad
# [6/9] Holographic error correction
#    d_holo=8.1905, p_thresh=0.010027, log_err=2.3412e-04
#    Resource overhead: {'n_logical_qubits': 10, 'n_physical_qubits': 490, ...}
# [7/9] Quantum algorithms
#    Shor(15): 3 x 5 = 15
#    Grover(n=8, M=2): k_opt=9, success=0.9310
#    VQE(Ising, n=4): E=-5.226251, err=3.21e-04
# [8/9] Machine learning pipeline
#    Final val_loss=0.421034, test_loss=0.438921
# [9/9] Trapped ions OTOC
#    lambda_L=8.0890e+05 s^-1, t_scramble=0.4985 ms, EE(25/50)=1.298393
#
# =================================================================
# All 9 pipeline stages completed successfully.
# =================================================================
#
# Full pipeline: PASS