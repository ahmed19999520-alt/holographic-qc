using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace HolographicQC.Algorithms
{
    public class GroverAlgorithm
    {
        private readonly int _n;
        private readonly int _N;
        private readonly HashSet<int> _targets;
        private readonly Random _rng;

        public GroverAlgorithm(int nQubits, IEnumerable<int> targets, int seed = 42)
        {
            _n = nQubits;
            _N = 1 << nQubits;
            _targets = new HashSet<int>(targets);
            _rng = new Random(seed);
        }

        public int OptimalIterations()
        {
            int M = _targets.Count;
            if (M == 0 || M >= _N) return 0;
            double theta = Math.Asin(Math.Sqrt((double)M / _N));
            return (int)Math.Floor(Math.PI / (4.0 * theta));
        }

        public double SuccessProbability(int nIterations)
        {
            int M = _targets.Count;
            if (M == 0) return 0.0;
            double theta = Math.Asin(Math.Sqrt((double)M / _N));
            double angle = (2 * nIterations + 1) * theta;
            return Math.Sin(angle) * Math.Sin(angle);
        }

        public double[] UniformSuperposition()
        {
            double amp = 1.0 / Math.Sqrt(_N);
            var state = new double[_N];
            for (int i = 0; i < _N; i++) state[i] = amp;
            return state;
        }

        public double[] ApplyOracle(double[] state)
        {
            var result = (double[])state.Clone();
            foreach (int t in _targets)
                if (t < _N) result[t] *= -1.0;
            return result;
        }

        public double[] ApplyDiffusion(double[] state)
        {
            double mean = state.Average();
            var result = new double[_N];
            for (int i = 0; i < _N; i++)
                result[i] = 2.0 * mean - state[i];
            return result;
        }

        public (double[] finalState, int iterations) Run(int? nIterations = null)
        {
            int k = nIterations ?? OptimalIterations();
            double[] state = UniformSuperposition();
            for (int i = 0; i < k; i++)
            {
                state = ApplyOracle(state);
                state = ApplyDiffusion(state);
            }
            return (state, k);
        }

        public Dictionary<string, object> RunWithMeasurement(int nShots = 1000, int? nIterations = null)
        {
            var (state, k) = Run(nIterations);
            double[] probs = state.Select(a => a * a).ToArray();
            double sum = probs.Sum();
            for (int i = 0; i < probs.Length; i++) probs[i] /= sum;

            var counts = new Dictionary<int, int>();
            for (int shot = 0; shot < nShots; shot++)
            {
                double u = _rng.NextDouble();
                double cumulative = 0.0;
                int outcome = _N - 1;
                for (int i = 0; i < _N; i++)
                {
                    cumulative += probs[i];
                    if (u <= cumulative) { outcome = i; break; }
                }
                if (!counts.ContainsKey(outcome)) counts[outcome] = 0;
                counts[outcome]++;
            }

            int successCount = _targets.Sum(t => counts.ContainsKey(t) ? counts[t] : 0);
            return new Dictionary<string, object>
            {
                ["state"] = state,
                ["probabilities"] = probs,
                ["measurements"] = counts,
                ["success_rate"] = (double)successCount / nShots,
                ["n_iterations"] = k,
                ["theoretical_success_prob"] = SuccessProbability(k),
            };
        }

        public Dictionary<string, object> CircuitResourceEstimate()
        {
            int k = OptimalIterations();
            double classical = (double)_N / Math.Max(1, _targets.Count);
            return new Dictionary<string, object>
            {
                ["n_qubits"] = _n,
                ["optimal_iterations"] = k,
                ["oracle_calls"] = k,
                ["total_gate_count"] = k * (_n + 1),
                ["classical_expected_calls"] = (int)classical,
                ["speedup_factor"] = classical / Math.Max(1, k),
            };
        }

        public bool VerifyQuadraticSpeedup()
        {
            int k = OptimalIterations();
            double classical = (double)_N / Math.Max(1, _targets.Count);
            double theoretical = Math.Sqrt((double)_N / Math.Max(1, _targets.Count));
            double speedup = classical / Math.Max(1, k);
            return Math.Abs(speedup - theoretical) / theoretical < 0.5;
        }
    }
}