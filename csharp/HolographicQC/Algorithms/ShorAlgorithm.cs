using System;
using System.Collections.Generic;
using System.Numerics;

namespace HolographicQC.Algorithms
{
    public class ShorAlgorithm
    {
        private readonly int _N;
        private readonly Random _rng;

        public ShorAlgorithm(int N, int seed = 42)
        {
            _N = N;
            _rng = new Random(seed);
        }

        public (int p, int q)? Factor(int maxAttempts = 100)
        {
            if (_N % 2 == 0)
                return (2, _N / 2);

            var primepower = IsPrimePower(_N);
            if (primepower.HasValue)
                return (primepower.Value.b, _N / primepower.Value.b);

            for (int attempt = 0; attempt < maxAttempts; attempt++)
            {
                int a = _rng.Next(2, _N);
                int g = Gcd(a, _N);
                if (g > 1 && g < _N)
                    return (g, _N / g);

                int? r = FindPeriodClassical(a);
                if (r == null || r.Value % 2 != 0)
                    continue;

                int x = (int)(ModPow(a, r.Value / 2, _N));
                if (x == _N - 1)
                    continue;

                int f1 = Gcd(x + 1, _N);
                int f2 = Gcd(x - 1, _N);

                if (f1 > 1 && f1 < _N)
                    return (f1, _N / f1);
                if (f2 > 1 && f2 < _N)
                    return (f2, _N / f2);
            }
            return null;
        }

        private int? FindPeriodClassical(int a)
        {
            int x = 1;
            for (int r = 1; r <= _N; r++)
            {
                x = (int)((long)x * a % _N);
                if (x == 1)
                    return r;
            }
            return null;
        }

        public bool VerifyFactorization(int p, int q)
            => p * q == _N && p > 1 && q > 1;

        public bool IsPrimeClassical(int n)
        {
            if (n < 2) return false;
            if (n == 2) return true;
            if (n % 2 == 0) return false;
            for (int i = 3; i * i <= n; i += 2)
                if (n % i == 0) return false;
            return true;
        }

        private (int b, int e)? IsPrimePower(int n)
        {
            int maxExp = (int)Math.Log2(n) + 1;
            for (int exp = 2; exp <= maxExp; exp++)
            {
                int b = (int)Math.Round(Math.Pow(n, 1.0 / exp));
                foreach (int candidate in new[] { b - 1, b, b + 1 })
                {
                    if (candidate > 1 && (long)Math.Pow(candidate, exp) == n)
                        return (candidate, exp);
                }
            }
            return null;
        }

        public static int Gcd(int a, int b)
        {
            while (b != 0) { int t = b; b = a % b; a = t; }
            return a;
        }

        public static long ModPow(long b, long exp, long mod)
        {
            long result = 1;
            b %= mod;
            while (exp > 0)
            {
                if ((exp & 1) == 1) result = result * b % mod;
                exp >>= 1;
                b = b * b % mod;
            }
            return result;
        }

        public Dictionary<string, int> CircuitResourceEstimate()
        {
            int n = (int)Math.Ceiling(Math.Log2(_N));
            int nPrecision = 2 * n;
            return new Dictionary<string, int>
            {
                ["n_logical_qubits"] = 2 * n + nPrecision,
                ["n_qft_gates"] = nPrecision * (nPrecision + 1) / 2,
                ["circuit_depth"] = nPrecision * n * n,
                ["n_modular_exp_gates"] = nPrecision * n * n,
            };
        }
    }
}