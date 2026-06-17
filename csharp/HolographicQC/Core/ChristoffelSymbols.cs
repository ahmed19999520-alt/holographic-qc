using System;

namespace HolographicQC.Core
{
    public class ChristoffelSymbols
    {
        private readonly int _dim;
        private readonly double[,] _g;
        private readonly double[,] _gInv;

        public ChristoffelSymbols(double[,] metric)
        {
            _dim = metric.GetLength(0);
            _g = (double[,])metric.Clone();
            _gInv = InvertMatrix(metric);
        }

        public static ChristoffelSymbols FromAds3Poincare(double adsRadius, double z)
        {
            double factor = adsRadius * adsRadius / (z * z);
            double[,] g = { { factor, 0, 0 }, { 0, factor, 0 }, { 0, 0, -factor } };
            return new ChristoffelSymbols(g);
        }

        public double[,,] Compute(double[,,] dg)
        {
            var gamma = new double[_dim, _dim, _dim];
            for (int sigma = 0; sigma < _dim; sigma++)
                for (int mu = 0; mu < _dim; mu++)
                    for (int nu = 0; nu < _dim; nu++)
                    {
                        double val = 0.0;
                        for (int lam = 0; lam < _dim; lam++)
                            val += 0.5 * _gInv[sigma, lam] * (
                                dg[nu, lam, mu] + dg[mu, lam, nu] - dg[lam, mu, nu]
                            );
                        gamma[sigma, mu, nu] = val;
                    }
            return gamma;
        }

        public double Ads3GeodesicLength(double x1, double z1, double x2, double z2, double adsRadius)
        {
            double chordal = ((x1 - x2) * (x1 - x2) + (z1 - z2) * (z1 - z2)) / (2.0 * z1 * z2);
            return adsRadius * Math.Acosh(1.0 + chordal);
        }

        public double Ads3GeodesicLengthBoundary(double ell, double uvCutoff, double adsRadius)
            => 2.0 * adsRadius * Math.Log(ell / uvCutoff);

        private static double[,] InvertMatrix(double[,] m)
        {
            int n = m.GetLength(0);
            double[,] aug = new double[n, 2 * n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++) aug[i, j] = m[i, j];
                aug[i, i + n] = 1.0;
            }
            for (int col = 0; col < n; col++)
            {
                int pivot = col;
                for (int row = col + 1; row < n; row++)
                    if (Math.Abs(aug[row, col]) > Math.Abs(aug[pivot, col])) pivot = row;
                for (int j = 0; j < 2 * n; j++)
                {
                    double tmp = aug[col, j]; aug[col, j] = aug[pivot, j]; aug[pivot, j] = tmp;
                }
                double diag = aug[col, col];
                if (Math.Abs(diag) < 1e-15) throw new InvalidOperationException("Singular matrix");
                for (int j = 0; j < 2 * n; j++) aug[col, j] /= diag;
                for (int row = 0; row < n; row++)
                {
                    if (row == col) continue;
                    double factor = aug[row, col];
                    for (int j = 0; j < 2 * n; j++) aug[row, j] -= factor * aug[col, j];
                }
            }
            double[,] inv = new double[n, n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) inv[i, j] = aug[i, j + n];
            return inv;
        }
    }
}