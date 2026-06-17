using System;
using HolographicQC.Core;
using HolographicQC.Algorithms;
using HolographicQC.Protection;

namespace HolographicQC
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("HolographicQC — Integration Test");
            Console.WriteLine(new string('=', 55));

            RunVirasoroTests();
            RunAdsCftTests();
            RunRyuTakayanagTests();
            RunChristoffelTests();
            RunShorTests();
            RunGroverTests();
            RunQFTTests();
            RunDecoherenceTests();
            RunMajoranaTests();
            RunHolographicCodeTests();

            Console.WriteLine("\nAll C# integration tests passed.");
        }

        static void RunVirasoroTests()
        {
            Console.WriteLine("\n[1/10] Virasoro Algebra");
            var alg = new VirasoroAlgebra(new VirasoroConfig { CentralCharge = 1.0, MaxMode = 8 });

            var (lin, cen) = alg.CommutatorScalar(3, -3);
            Assert(Math.Abs(lin - 6.0) < 1e-10, $"[L_3,L_-3] linear={lin}");
            Assert(Math.Abs(cen - 2.0) < 1e-10, $"[L_3,L_-3] central={cen}");

            bool jacobi = alg.VerifyJacobiIdentity(1, 2, -3);
            Assert(jacobi, "Jacobi identity (1,2,-3)");

            double chi = alg.Character(0.0, 0.1, 30);
            Assert(chi > 0, $"Character chi(h=0)={chi}");

            double[,] kac = alg.KacTable(4, 3);
            Assert(kac.GetLength(0) == 3 && kac.GetLength(1) == 2, "Kac table shape");

            double lam = alg.LyapunovExponent(4.0);
            const double kB = 1.380649e-23, hbar = 1.054571817e-34;
            double bound = 2.0 * Math.PI * kB * 4.0 / hbar;
            Assert(lam <= bound + 1e-10, $"lambda_L={lam:.4e} <= bound={bound:.4e}");

            Console.WriteLine($"   c=1: [L3,L-3]=(6,2), Jacobi=OK, chi={chi:.4f}, lambda_L/bound={lam/bound:.6f}");
        }

        static void RunAdsCftTests()
        {
            Console.WriteLine("\n[2/10] AdS3/CFT2");
            var ads = new AdsCft3(1.0, 1.1e-9, 5e5);
            double c_bh = ads.CentralCharge;
            double sigma = ads.OpticalConductivityDC();
            double wf = ads.WiedemannFranzRatio();
            double lam = ads.LyapunovExponent(4.0);
            double enh = ads.HolographicCoherenceEnhancement(1e-6, 1.1e-9);

            Assert(Math.Abs(c_bh - 1.0) < 1e-6, $"c={c_bh}");
            Assert(sigma > 0, $"sigma_0={sigma}");
            Assert(Math.Abs(wf) < 1e-30, $"WF ratio={wf} (expected 0 for c=1)");
            Assert(enh > 1.0, $"enhancement={enh}");

            double delta_m0 = ads.ScalingDimensionFromMass(0.0);
            Assert(Math.Abs(delta_m0 - 1.5) < 1e-10, $"Delta(m=0)={delta_m0}");

            Console.WriteLine($"   c={c_bh}, sigma={sigma:.4e}, WF={wf:.2e}, enh={enh:.4f}, Delta={delta_m0}");
        }

        static void RunRyuTakayanagTests()
        {
            Console.WriteLine("\n[3/10] Ryu-Takayanagi");
            var rt = new RyuTakayanagi(new RTConfig
            {
                CentralCharge = 1.0,
                NewtonConstant3D = 1.1e-9,
                AdsRadius = 1.1e-9,
                UVCutoff = 0.3e-9
            });

            double S = rt.EntanglementEntropyCentralCharge(100e-9);
            double S_expected = (1.0 / 3.0) * Math.Log(100e-9 / 0.3e-9);
            Assert(Math.Abs(S - S_expected) < 1e-10, $"S_A={S:.6f}");

            double MI = rt.MutualInformation(50e-9, 50e-9, 10e-9);
            Assert(MI >= -1e-10, $"MI={MI}");

            double S2 = rt.RenyiEntropy(100e-9, 2);
            double S3 = rt.RenyiEntropy(100e-9, 3);
            Assert(S2 >= S3, "S2 >= S3");

            double d_holo = rt.HolographicCodeDistance(7.0, 1e-6, 1e-9);
            Assert(d_holo > 7.0, $"d_holo={d_holo}");

            Console.WriteLine($"   S_A={S:.6f}, MI={MI:.6f}, S2={S2:.6f}, d_holo={d_holo:.4f}");
        }

        static void RunChristoffelTests()
        {
            Console.WriteLine("\n[4/10] Christoffel Symbols");
            var cs = ChristoffelSymbols.FromAds3Poincare(1.1e-9, 1e-10);
            double L_geo = cs.Ads3GeodesicLengthBoundary(100e-9, 0.3e-9, 1.1e-9);
            double expected = 2.0 * 1.1e-9 * Math.Log(100e-9 / 0.3e-9);
            Assert(Math.Abs(L_geo - expected) < 1e-25, $"L_geo={L_geo}");
            Console.WriteLine($"   L_gamma(100nm)={L_geo:.6e}");
        }

        static void RunShorTests()
        {
            Console.WriteLine("\n[5/10] Shor's Algorithm");
            int[] testNs = { 6, 15, 21, 35, 77 };
            foreach (int N in testNs)
            {
                var shor = new ShorAlgorithm(N, seed: 42);
                var result = shor.Factor(maxAttempts: 100);
                Assert(result.HasValue, $"Shor({N}) returned result");
                if (result.HasValue)
                {
                    var (p, q) = result.Value;
                    Assert(shor.VerifyFactorization(p, q), $"{p} x {q} = {N}");
                    Console.WriteLine($"   N={N}: {p} x {q} = {p*q}  OK");
                }
            }
        }

        static void RunGroverTests()
        {
            Console.WriteLine("\n[6/10] Grover's Algorithm");
            var configs = new (int n, int[] t)[] {
                (4, new[]{5}), (6, new[]{42}), (8, new[]{100,200})
            };
            foreach (var (n, targets) in configs)
            {
                var grover = new GroverAlgorithm(n, targets, seed: 0);
                int k = grover.OptimalIterations();
                double p = grover.SuccessProbability(k);
                var result = grover.RunWithMeasurement(nShots: 500);
                double p_meas = (double)result["success_rate"];
                bool speedup = grover.VerifyQuadraticSpeedup();
                Assert(p > 0.8, $"Grover(n={n}): P_theory={p:.4f}");
                Console.WriteLine($"   n={n}, M={targets.Length}: k={k}, P_th={p:.4f}, P_meas={p_meas:.4f}, speedup={speedup}");
            }
        }

        static void RunQFTTests()
        {
            Console.WriteLine("\n[7/10] Quantum Fourier Transform");
            foreach (int n in new[] { 2, 3, 4, 5, 6 })
            {
                var qft = new QuantumFourierTransform(n);
                bool unitary = qft.VerifyUnitarity();
                Assert(unitary, $"QFT unitary n={n}");
                Console.WriteLine($"   n={n}: depth={qft.CircuitDepth()}, gates={qft.GateCount()}, unitary={unitary}");
            }
        }

        static void RunDecoherenceTests()
        {
            Console.WriteLine("\n[8/10] Holographic Decoherence");
            var cfg = new DecoherenceConfig
            {
                Temperature = 4.0,
                PhononCoupling = 2.1e-3,
                SystemSize = 1e-6,
                CoherenceLength = 1.1e-9,
                CentralCharge = 1.0
            };
            var model = new HolographicDecoherenceModel(cfg);
            var report = model.FullReport(4.0);
            Assert(report["T2_holo_ns"] > report["T2_std_ns"], "T2_holo > T2_std");
            Assert(report["enhancement_factor"] > 1.0, "enhancement > 1");
            Assert(report["quantum_fisher_info"] > 1.0, "QFI > 1");
            Console.WriteLine($"   T2_std={report["T2_std_ns"]:.4f}ns, T2_holo={report["T2_holo_ns"]:.4f}ns, enh={report["enhancement_factor"]:.4f}");
        }

        static void RunMajoranaTests()
        {
            Console.WriteLine("\n[9/10] Majorana Qubit");
            var cfg = new MajoranaQubitConfig
            {
                WireLength = 100e-9,
                CoherenceLength = 1.1e-9,
                FermiVelocity = 5e5,
                CentralCharge = 1.0
            };
            var qubit = new MajoranaQubit(cfg);
            double F = qubit.GateFidelity(0.01);
            double phi = qubit.HolographicBerryPhase();
            double enh = qubit.HolographicEnhancementFactor();
            double T2 = qubit.TotalCoherenceTime(1e-6);
            Assert(F > 0.0 && F < 1.0, $"Fidelity={F}");
            Assert(enh > 1.0, $"Enhancement={enh}");
            Console.WriteLine($"   F={F:.8f}, phi_Berry={phi:.6f}, enh={enh:.4f}, T2={T2:.4e}s");
        }

        static void RunHolographicCodeTests()
        {
            Console.WriteLine("\n[10/10] Holographic Error Correction");
            var sc = new SurfaceCode(7);
            double d_holo = sc.HolographicCodeDistance(1.0, 1e-6, 1e-9);
            double p_thresh = sc.HolographicThreshold(1.0, 0.01);
            double log_err = sc.LogicalErrorRate(0.005);
            Assert(d_holo > 7.0, $"d_holo={d_holo}");
            Assert(p_thresh > 0.01, $"p_thresh={p_thresh}");
            Console.WriteLine($"   d={sc.Distance}, d_holo={d_holo:.4f}, p_thresh={p_thresh:.6f}, log_err={log_err:.4e}");

            var hc = new HolographicCode("pentagon", 1.0);
            double rate = hc.EncodingRate();
            Assert(rate > 0 && rate < 1, $"encoding_rate={rate}");
            Console.WriteLine($"   Pentagon code: encoding_rate={rate:.4f}");
        }

        static void Assert(bool condition, string message)
        {
            if (!condition)
                throw new Exception($"Assertion failed: {message}");
        }
    }
}