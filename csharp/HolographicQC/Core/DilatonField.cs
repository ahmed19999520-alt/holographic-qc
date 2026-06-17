using System;

namespace HolographicQC.Core
{
    public class DilatonConfig
    {
        public double AdsRadius { get; set; }
        public double MassSqTimesLSq { get; set; } = 0.0;
        public double FermiVelocity { get; set; } = 5e5;
        public double LuttingerParameter { get; set; } = 1.0;
    }

    public class DilatonField
    {
        private readonly DilatonConfig _cfg;
        public double Nu { get; }
        public double Delta { get; }

        public DilatonField(DilatonConfig config)
        {
            _cfg = config;
            Nu = Math.Sqrt(0.25 + config.MassSqTimesLSq);
            Delta = 1.0 + Nu;
        }

        public double TwoPointFunction(double x, double xPrime)
        {
            double sep = Math.Abs(x - xPrime);
            if (sep < 1e-15) throw new ArgumentException("Coincident points");
            return 0.5 / Math.Pow(sep, 2 * Delta);
        }

        public double HolographicCorrelationWithLogCorrection(
            double x, double xi, double c, double A = 1.0)
        {
            double powerLaw = A / (x * x);
            double logCorr = (c / (12.0 * Math.PI * Math.PI)) * Math.Log(Math.Abs(x) / xi);
            return powerLaw * (1.0 + logCorr);
        }

        public double DynamicalStructureFactor(
            double q, double omega, double temperature, double A = 1.0)
        {
            const double kB = 1.380649e-23;
            const double hbar = 1.054571817e-34;
            double T_nat = kB * temperature / hbar;
            double threshold = _cfg.FermiVelocity * Math.Abs(q);
            if (omega <= threshold) return 0.0;
            double numerator = A / Math.Sqrt(omega * omega - threshold * threshold);
            double bose = 1.0 / (1.0 - Math.Exp(-omega / T_nat));
            return numerator * bose;
        }

        public double NoisePowerAfterScreening(
            double systemSize, double coherenceLength, double deltaN, double noisePower)
        {
            double screening = Math.Pow(coherenceLength / systemSize, 4.0 * deltaN) / 5.0;
            return noisePower * screening;
        }

        public double OpticalConductivity(double omega, double sigma0, double omega0)
            => sigma0 * (1.0 + 0.1 * omega / omega0);
    }
}