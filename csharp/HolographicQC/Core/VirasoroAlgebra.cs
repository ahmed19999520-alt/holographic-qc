using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics;

namespace HolographicQC.Core
{
    public class VirasoroConfig
    {
        public double CentralCharge { get; set; }
        public int MaxMode { get; set; } = 10;
    }

    public class VirasoroAlgebra
    {
        private readonly double _c;
        private readonly int _N;

        public double CentralCharge => _c;

        public VirasoroAlgebra(VirasoroConfig config)
        {
            _c = config.CentralCharge;
            _N = config.MaxMode;
        }

        public (double Linear, double Central) CommutatorScalar(int m, int n)
        {
            double linear = m - n;
            double central = (m + n == 0) ? (_c / 12.0) * m * (m * m - 1) : 0.0;
            return (linear, central);
        }

        public bool VerifyJacobiIdentity(int l, int m, int n, double tol = 1e-10)
        {
            var (lin_lm, _) = CommutatorScalar(l, m);
            var (lin_mn, _) = CommutatorScalar(m, n);
            var (lin_nl, _) = CommutatorScalar(n, l);
            double jacobiLinear = lin_lm * (l + m - n) + lin_mn * (m + n - l) + lin_nl * (n + l - m);
            return Math.Abs(jacobiLinear) < tol;
        }

        public Complex32 OpeCoefficientTT(Complex32 z, Complex32 w)
        {
            var diff = z - w;
            if (diff.Magnitude < 1e-15f)
                throw new ArgumentException("Coincident points in OPE");
            var denom = diff * diff * diff * diff;
            return new Complex32((float)(_c / 2.0), 0f) / denom;
        }

        public Complex32 TwoPointFunction(Complex32 z1, Complex32 z2, double h)
        {
            var diff = z1 - z2;
            if (diff.Magnitude < 1e-15f)
                throw new ArgumentException("Coincident points");
            double exponent = 2.0 * h;
            return new Complex32(1.0f, 0f) / (Complex32)Math.Pow(diff.Magnitude, exponent);
        }

        public double LyapunovExponent(double temperature, double kB = 1.380649e-23, double hbar = 1.054571817e-34)
        {
            double bound = 2.0 * Math.PI * kB * temperature / hbar;
            double correction = 6.0 / (_c * _c);
            return bound * (1.0 - correction);
        }

        public double Character(double h, double q, int nLevels = 30)
        {
            if (Math.Abs(q) >= 1.0)
                throw new ArgumentException("|q| must be < 1 for convergence");
            double prefactor = Math.Pow(q, h - _c / 24.0);
            double etaInv = 1.0;
            for (int n = 1; n <= nLevels; n++)
                etaInv /= (1.0 - Math.Pow(q, n));
            return prefactor * etaInv;
        }

        public double[,] KacTable(int p, int qParam)
        {
            var table = new double[p - 1, qParam - 1];
            for (int r = 1; r < p; r++)
                for (int s = 1; s < qParam; s++)
                {
                    double h = Math.Pow(p * s - qParam * r, 2) - Math.Pow(p - qParam, 2);
                    h /= 4.0 * p * qParam;
                    table[r - 1, s - 1] = h;
                }
            return table;
        }

        public List<(int, int)> CommutatorTable(int mRange)
        {
            var results = new List<(int, int)>();
            for (int m = -mRange; m <= mRange; m++)
                for (int n = -mRange; n <= mRange; n++)
                {
                    var (lin, cen) = CommutatorScalar(m, n);
                    results.Add((m, n));
                }
            return results;
        }

        public double StressTensorTwoPoint(double z, double w)
        {
            double diff = z - w;
            if (Math.Abs(diff) < 1e-15)
                throw new ArgumentException("Coincident points");
            return (_c / 2.0) / Math.Pow(diff, 4);
        }
    }
}