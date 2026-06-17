using System;
using System.Numerics;

namespace HolographicQC.Protection
{
    public class MajoranaQubitConfig
    {
        public double WireLength { get; set; }
        public double CoherenceLength { get; set; }
        public double FermiVelocity { get; set; } = 5e5;
        public double CentralCharge { get; set; } = 1.0;
    }

    public class MajoranaQubit
    {
        private readonly MajoranaQubitConfig _config;
        private static readonly double[,] SigmaZ = { { 1, 0 }, { 0, -1 } };

        public MajoranaQubit(MajoranaQubitConfig config)
        {
            _config = config;
        }

        public double[,] ParityOperator()
        {
            return new double[,] { { 0, -1 }, { 1, 0 } };
        }

        public Complex[,] BraidingUnitary()
        {
            double angle = Math.PI / 4.0;
            double c = Math.Cos(angle);
            double s = Math.Sin(angle);
            return new Complex[,]
            {
                { new Complex(c, 0), new Complex(0, -s) },
                { new Complex(0, -s), new Complex(c, 0) }
            };
        }

        public double HolographicBerryPhase()
        {
            double L = _config.WireLength;
            double xi = _config.CoherenceLength;
            double correction = (xi / L) * Math.Log(L / xi);
            return (Math.PI / 4.0) * (1.0 + correction);
        }

        public double GateFidelity(double epsilon0)
        {
            double L = _config.WireLength;
            double xi = _config.CoherenceLength;
            double c = _config.CentralCharge;
            return 1.0 - epsilon0 * Math.Pow(L / xi, -c / 6.0);
        }

        public double TopologicalCoherenceTime(double tau0)
        {
            double L = _config.WireLength;
            double xi = _config.CoherenceLength;
            return tau0 * Math.Exp(L / xi);
        }

        public double HolographicEnhancementFactor()
        {
            double L = _config.WireLength;
            double xi = _config.CoherenceLength;
            double c = _config.CentralCharge;
            return Math.Pow(L / xi, c / 6.0);
        }

        public double TotalCoherenceTime(double tau0)
        {
            return TopologicalCoherenceTime(tau0) * HolographicEnhancementFactor();
        }

        public double EnergySplitting()
        {
            double L = _config.WireLength;
            double xi = _config.CoherenceLength;
            double hbar = 1.054571817e-34;
            double overlap = Math.Exp(-L / xi);
            return hbar * _config.FermiVelocity * overlap / xi;
        }

        public double ParityFlipRate(double gamma0)
        {
            double L = _config.WireLength;
            double xi = _config.CoherenceLength;
            double c = _config.CentralCharge;
            return gamma0 * Math.Exp(-c * L / (6.0 * Math.PI * xi));
        }
    }
}