using System;
using System.Collections.Generic;

namespace HolographicQC.Protection
{
    public class DecoherenceConfig
    {
        public double Temperature { get; set; }
        public double PhononCoupling { get; set; } = 2.1e-3;
        public double SystemSize { get; set; } = 1e-6;
        public double CoherenceLength { get; set; } = 1.1e-9;
        public double CentralCharge { get; set; } = 1.0;
    }

    public class HolographicDecoherenceModel
    {
        private const double KBoltzmann = 1.380649e-23;
        private const double HBar = 1.054571817e-34;
        private readonly DecoherenceConfig _config;

        public HolographicDecoherenceModel(DecoherenceConfig config)
        {
            _config = config;
        }

        public double StandardPhononRate(double temperature)
        {
            double kT_over_hbar = KBoltzmann * temperature / HBar;
            return _config.PhononCoupling * kT_over_hbar * kT_over_hbar;
        }

        public double CoherenceTimeStandard(double temperature)
        {
            double rate = StandardPhononRate(temperature);
            return rate > 0 ? 1.0 / rate : double.PositiveInfinity;
        }

        public double CoherenceTimeRatio()
        {
            double ratio = _config.SystemSize / _config.CoherenceLength;
            if (ratio <= 0) throw new ArgumentException("SystemSize must exceed CoherenceLength");
            return Math.Pow(ratio, _config.CentralCharge / 6.0);
        }

        public double CoherenceTimeHolographic(double temperature)
        {
            double T2_std = CoherenceTimeStandard(temperature);
            return T2_std * CoherenceTimeRatio();
        }

        public double HolographicDecoherenceRate(double temperature)
        {
            double gamma_std = StandardPhononRate(temperature);
            double xi = _config.CoherenceLength;
            double L = _config.SystemSize;
            double screening = Math.Pow(xi / L, 4.0);
            return gamma_std * screening / 5.0;
        }

        public double CombinedTopologicalHolographicT2(double T2_bare)
        {
            double L = _config.SystemSize;
            double xi = _config.CoherenceLength;
            double topo_factor = Math.Exp(L / xi);
            double holo_factor = Math.Pow(L / xi, _config.CentralCharge / 6.0);
            return T2_bare * topo_factor * holo_factor;
        }

        public double QuantumFisherInformation()
        {
            double L = _config.SystemSize;
            double xi = _config.CoherenceLength;
            double c = _config.CentralCharge;
            return 1.0 + (c / 6.0) * Math.Log(L / xi);
        }

        public Dictionary<string, double> FullReport(double temperature)
        {
            return new Dictionary<string, double>
            {
                ["gamma_std"] = StandardPhononRate(temperature),
                ["gamma_holo"] = HolographicDecoherenceRate(temperature),
                ["T2_std_ns"] = CoherenceTimeStandard(temperature) * 1e9,
                ["T2_holo_ns"] = CoherenceTimeHolographic(temperature) * 1e9,
                ["enhancement_factor"] = CoherenceTimeRatio(),
                ["quantum_fisher_info"] = QuantumFisherInformation(),
            };
        }
    }
}