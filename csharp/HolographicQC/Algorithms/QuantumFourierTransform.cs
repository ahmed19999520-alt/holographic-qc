using System;
using System.Numerics;

namespace HolographicQC.Algorithms
{
    public class QuantumFourierTransform
    {
        private readonly int _n;
        private readonly int _N;
        private Complex[,] _matrix;

        public int NQubits => _n;
        public int Dimension => _N;

        public QuantumFourierTransform(int nQubits)
        {
            _n = nQubits;
            _N = 1 << nQubits;
        }

        public Complex[,] Matrix()
        {
            if (_matrix != null) return _matrix;
            _matrix = new Complex[_N, _N];
            double sqrtN = Math.Sqrt(_N);
            for (int j = 0; j < _N; j++)
                for (int k = 0; k < _N; k++)
                {
                    double angle = 2.0 * Math.PI * j * k / _N;
                    _matrix[j, k] = new Complex(Math.Cos(angle), Math.Sin(angle)) / sqrtN;
                }
            return _matrix;
        }

        public Complex[] Apply(Complex[] state)
        {
            if (state.Length != _N)
                throw new ArgumentException($"State must have length {_N}");
            var F = Matrix();
            var result = new Complex[_N];
            for (int j = 0; j < _N; j++)
            {
                result[j] = Complex.Zero;
                for (int k = 0; k < _N; k++)
                    result[j] += F[j, k] * state[k];
            }
            return result;
        }

        public Complex[] ApplyInverse(Complex[] state)
        {
            if (state.Length != _N)
                throw new ArgumentException($"State must have length {_N}");
            var F = Matrix();
            var result = new Complex[_N];
            for (int j = 0; j < _N; j++)
            {
                result[j] = Complex.Zero;
                for (int k = 0; k < _N; k++)
                    result[j] += Complex.Conjugate(F[k, j]) * state[k];
            }
            return result;
        }

        public bool VerifyUnitarity(double tol = 1e-10)
        {
            var F = Matrix();
            for (int i = 0; i < _N; i++)
                for (int j = 0; j < _N; j++)
                {
                    Complex sum = Complex.Zero;
                    for (int k = 0; k < _N; k++)
                        sum += F[i, k] * Complex.Conjugate(F[j, k]);
                    double expected = (i == j) ? 1.0 : 0.0;
                    if (Math.Abs(sum.Real - expected) > tol || Math.Abs(sum.Imaginary) > tol)
                        return false;
                }
            return true;
        }

        public double EstimatePhase(Complex[] state)
        {
            var freqState = ApplyInverse(state);
            int bestK = 0;
            double bestProb = 0.0;
            for (int k = 0; k < _N; k++)
            {
                double prob = freqState[k].Magnitude * freqState[k].Magnitude;
                if (prob > bestProb) { bestProb = prob; bestK = k; }
            }
            return (double)bestK / _N;
        }

        public int CircuitDepth() => _n * (_n + 1) / 2;
        public int GateCount() => _n + _n * (_n - 1) / 2;

        public Complex[] PhaseEstimationState(double phase)
        {
            var state = new Complex[_N];
            for (int k = 0; k < _N; k++)
            {
                double angle = 2.0 * Math.PI * k * phase;
                state[k] = new Complex(Math.Cos(angle), Math.Sin(angle)) / Math.Sqrt(_N);
            }
            return state;
        }

        public Complex[,] RotationGate(int k)
        {
            double angle = 2.0 * Math.PI / Math.Pow(2, k);
            return new Complex[,]
            {
                { Complex.One, Complex.Zero },
                { Complex.Zero, new Complex(Math.Cos(angle), Math.Sin(angle)) }
            };
        }
    }
}