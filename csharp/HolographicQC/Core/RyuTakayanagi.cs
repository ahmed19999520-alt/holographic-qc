using System;

namespace HolographicQC.Core
{
    public class RTConfig
    {
        public double CentralCharge { get; set; }
        public double NewtonConstant3D { get; set; }
        public double AdsRadius { get; set; }
        public double UVCutoff { get; set; } = 1e-10;
    }

    public class RyuTakayanagi
    {
        private readonly RTConfig _config;

        public double C => _config.CentralCharge;
        public double G3 => _config.NewtonConstant3D;
        public double L => _config.AdsRadius;
        public double A => _config.UVCutoff;

        public RyuTakayanagi(RTConfig config)
        {
            _config = config;
        }

        public double GeodesicLength(double intervalLength)
        {
            if (intervalLength <= A) throw new ArgumentException("Interval must exceed UV cutoff");
            return 2.0 * L * Math.Log(intervalLength / A);
        }

        public double EntanglementEntropy(double intervalLength)
            => GeodesicLength(intervalLength) / (4.0 * G3);

        public double EntanglementEntropyCentralCharge(double intervalLength)
            => (C / 3.0) * Math.Log(intervalLength / A);

        public double RenyiEntropy(double intervalLength, int n)
        {
            if (n == 1) return EntanglementEntropyCentralCharge(intervalLength);
            return C * (n + 1) / (6.0 * n) * Math.Log(intervalLength / A);
        }

        public double MutualInformation(double l1, double l2, double sep)
        {
            double S_A = EntanglementEntropyCentralCharge(l1);
            double S_B = EntanglementEntropyCentralCharge(l2);
            double phase1 = S_A + S_B;
            double phase2 = EntanglementEntropyCentralCharge(l1 + sep + l2)
                          + EntanglementEntropyCentralCharge(sep);
            return S_A + S_B - Math.Min(phase1, phase2);
        }

        public double HolographicCodeDistance(double dStd, double systemSize, double latticeSpacing)
        {
            double correction = (C / (6.0 * Math.PI)) * Math.Log(systemSize / latticeSpacing);
            return dStd * (1.0 + correction);
        }

        public double ErrorThresholdHolographic(double pThreshStd)
            => pThreshStd * (1.0 + C / (12.0 * Math.PI));

        public double EntanglementEntropyFiniteTemperature(double intervalLength, double temperature)
        {
            const double kB = 1.380649e-23;
            const double hbar = 1.054571817e-34;
            double beta = hbar / (kB * temperature);
            double betaOverPi = beta / Math.PI;
            double arg = betaOverPi * Math.Sinh(Math.PI * intervalLength / beta);
            return (C / 3.0) * Math.Log(arg / A);
        }
    }
}