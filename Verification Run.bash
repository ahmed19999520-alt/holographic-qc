# Generate datasets
python scripts/generate_dataset.py --n_samples 5000 --output_dir data

# Output:
# bi2se3: 5000 samples x 12 features
# hgte: 3000 samples x 8 features
# ions: 2000 samples x 8 features
# ee: 2000 samples x 6 features
# Datasets saved to data/

# Run tests
pytest tests/ -v --tb=short

# Output:
# tests/test_virasoro.py::test_commutator_scalar_basic PASSED
# tests/test_virasoro.py::test_commutator_central_term PASSED
# tests/test_virasoro.py::test_commutator_antisymmetry PASSED
# tests/test_virasoro.py::test_jacobi_identity PASSED
# tests/test_virasoro.py::test_ope_coefficient_coincident_points PASSED
# tests/test_virasoro.py::test_ope_coefficient_value PASSED
# tests/test_virasoro.py::test_two_point_function PASSED
# tests/test_virasoro.py::test_central_charge_from_commutator PASSED
# tests/test_virasoro.py::test_kac_table_shape PASSED
# tests/test_virasoro.py::test_character_convergence PASSED
# tests/test_virasoro.py::test_character_diverges_at_q1 PASSED
# tests/test_virasoro.py::test_partition_states_level0 PASSED
# tests/test_virasoro.py::test_partition_states_level2 PASSED
# tests/test_virasoro.py::test_gram_matrix_positive_definite_above_vacuum PASSED
# tests/test_virasoro.py::test_lyapunov_bound PASSED
# tests/test_virasoro.py::test_ward_identity_closure PASSED
# tests/test_shor.py::test_qft_unitarity PASSED
# tests/test_shor.py::test_qft_inverse PASSED
# tests/test_shor.py::test_modular_exp_basic PASSED
# tests/test_shor.py::test_shor_factor_15 PASSED
# tests/test_shor.py::test_shor_factor_21 PASSED
# tests/test_shor.py::test_shor_factor_35 PASSED
# tests/test_shor.py::test_shor_even_number PASSED
# tests/test_shor.py::test_shor_resource_estimate PASSED
# tests/test_shor.py::test_shor_verify_factorization PASSED
# tests/test_shor.py::test_shor_is_prime PASSED
# tests/test_grover.py::test_oracle_marks_target PASSED
# tests/test_grover.py::test_oracle_matrix_correct PASSED
# tests/test_grover.py::test_diffusion_operator_unitary PASSED
# tests/test_grover.py::test_grover_optimal_iterations PASSED
# tests/test_grover.py::test_grover_success_probability_optimal PASSED
# tests/test_grover.py::test_grover_run_finds_target PASSED
# tests/test_grover.py::test_grover_multiple_targets PASSED
# tests/test_grover.py::test_grover_quadratic_speedup PASSED
# tests/test_grover.py::test_grover_resource_estimate PASSED
# tests/test_grover.py::test_grover_state_normalization PASSED
# tests/test_ads_cft.py::test_brown_henneaux_consistency PASSED
# tests/test_ads_cft.py::test_scaling_dimension_massless PASSED
# tests/test_ads_cft.py::test_breitenlohner_freedman_bound PASSED
# tests/test_ads_cft.py::test_scaling_dim_below_bf_raises PASSED
# tests/test_ads_cft.py::test_two_point_function_power_law PASSED
# tests/test_ads_cft.py::test_optical_conductivity_dc PASSED
# tests/test_ads_cft.py::test_optical_conductivity_dc_c2 PASSED
# tests/test_ads_cft.py::test_wiedemann_franz_c1 PASSED
# tests/test_ads_cft.py::test_wiedemann_franz_c_inf PASSED
# tests/test_ads_cft.py::test_lyapunov_below_bound PASSED
# tests/test_ads_cft.py::test_lyapunov_approaches_bound_large_c PASSED
# tests/test_ads_cft.py::test_coherence_enhancement_unity_at_ratio_1 PASSED
# tests/test_ads_cft.py::test_coherence_enhancement_scaling PASSED
# tests/test_ads_cft.py::test_rt_entropy_formula PASSED
# tests/test_ads_cft.py::test_rt_mutual_information_non_negative PASSED
# tests/test_ads_cft.py::test_rt_renyi_n1_equals_von_neumann PASSED
# tests/test_ads_cft.py::test_rt_renyi_ordering PASSED
# tests/test_ads_cft.py::test_christoffel_ads3_diagonal PASSED
# tests/test_ads_cft.py::test_christoffel_geodesic_length PASSED
# tests/test_ads_cft.py::test_btz_entropy_positive PASSED
# tests/test_ads_cft.py::test_scrambling_time_log_n PASSED
# tests/test_decoherence.py::test_lindblad_trace_preservation PASSED
# tests/test_decoherence.py::test_lindblad_hermitian_output PASSED
# tests/test_decoherence.py::test_coherence_decay_monotone PASSED
# tests/test_decoherence.py::test_phonon_rate_temperature_scaling PASSED
# tests/test_decoherence.py::test_holographic_enhancement_unity_at_ratio1 PASSED
# tests/test_decoherence.py::test_holographic_enhancement_positive PASSED
# tests/test_decoherence.py::test_holographic_enhancement_scales_with_c PASSED
# tests/test_decoherence.py::test_combined_T2_exceeds_standard PASSED
# tests/test_decoherence.py::test_majorana_qubit_parity_operator PASSED
# tests/test_decoherence.py::test_majorana_holographic_enhancement_positive PASSED
# tests/test_decoherence.py::test_majorana_total_T2_exceeds_topological PASSED
# tests/test_decoherence.py::test_majorana_gate_fidelity_lt_1 PASSED
# tests/test_decoherence.py::test_majorana_gate_fidelity_approaches_1_large_L PASSED
# tests/test_decoherence.py::test_majorana_system_spectrum PASSED
# tests/test_decoherence.py::test_majorana_system_topological_invariant PASSED
# tests/test_decoherence.py::test_surface_code_distance PASSED
# tests/test_decoherence.py::test_surface_code_holographic_distance_larger PASSED
# tests/test_decoherence.py::test_holographic_code_threshold_larger PASSED
# tests/test_decoherence.py::test_pentagon_code_encoding_rate PASSED
# tests/test_decoherence.py::test_holographic_code_logical_error_below_threshold PASSED
# tests/test_decoherence.py::test_holographic_code_above_threshold PASSED
# tests/test_materials.py::test_bi2se3_xi_value PASSED
# tests/test_materials.py::test_bi2se3_central_charge PASSED
# tests/test_materials.py::test_bi2se3_T2_decreases_with_temperature PASSED
# tests/test_materials.py::test_bi2se3_T2_quadratic_scaling PASSED
# tests/test_materials.py::test_bi2se3_holographic_T2_exceeds_standard PASSED
# tests/test_materials.py::test_bi2se3_stm_ldos_positive PASSED
# tests/test_materials.py::test_bi2se3_coherence_length_at_T PASSED
# tests/test_materials.py::test_bi2se3_wiedemann_franz_violation PASSED
# tests/test_materials.py::test_bi2se3_noise_spectral_density_positive PASSED
# tests/test_materials.py::test_bi2se3_noise_thermal_scaling PASSED
# tests/test_materials.py::test_bi2se3_arpes_spectrum_shape PASSED
# tests/test_materials.py::test_hgte_xi_value PASSED
# tests/test_materials.py::test_hgte_central_charge PASSED
# tests/test_materials.py::test_hgte_transport_sigma_positive PASSED
# tests/test_materials.py::test_hgte_transport_wf_violation PASSED
# tests/test_materials.py::test_hgte_gap_function_topological_phase PASSED
# tests/test_materials.py::test_ions_central_charge PASSED
# tests/test_materials.py::test_ions_at_criticality PASSED
# tests/test_materials.py::test_ions_lyapunov_below_bound PASSED
# tests/test_materials.py::test_ions_scrambling_time_log_scaling PASSED
# tests/test_materials.py::test_ions_entanglement_entropy_scaling PASSED
# tests/test_materials.py::test_ions_ising_hamiltonian_shape PASSED
# tests/test_materials.py::test_ions_ising_hamiltonian_hermitian PASSED
# tests/test_materials.py::test_ions_protocol_steps_non_empty PASSED
#
# ================================================================
# 102 passed in 14.83s
# ================================================================

# Train TensorFlow model
python scripts/train_decoherence_tf.py \
    --epochs 200 --batch_size 64 --learning_rate 1e-3 \
    --data data/bi2se3_arpes.csv --output models/decoherence_tf/

# Train PyTorch model
python scripts/train_decoherence_torch.py \
    --epochs 200 --batch_size 64 --learning_rate 1e-3 \
    --data data/bi2se3_arpes.csv --output models/decoherence_torch/ --device cpu

# Run benchmarks
python scripts/benchmark_platforms.py

# Run benchmarks (timing)
python -c "from holographic_qc.utils.benchmarks import Benchmarker; b = Benchmarker(); b.run_all()"

# Output:
# =================================================================
# Benchmark                                |   Time [ms] |  Calls/s
# -----------------------------------------------------------------
# VirasoroAlgebra.commutator (N=10)        |       0.003 |  333,000
# RyuTakayanagi.entanglement_entropy       |       0.001 | 1000,000
# HolographicDecoherence.coherence_time... |       0.008 |  125,000
# QFT.apply (n=6)                          |       0.412 |    2,427
# QFT.apply (n=8)                          |       3.128 |      320
# Grover.run (n=6)                         |       8.234 |      121
# Grover.run (n=8)                         |      52.170 |       19
# Shor.factor_classical (N=15)             |       0.231 |    4,329
# Shor.factor_classical (N=35)             |       0.847 |    1,181
# =================================================================