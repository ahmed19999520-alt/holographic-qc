using System;

namespace HolographicQC.Core
{
    public class AdsCft3
    {
        public double CentralCharge { get; }
        public double AdsRadius { get; }
        public double NewtonConstant3D { get; }
        public double FermiVelocity { get; }

        private const double KBoltzmann = 1.380649e-23;
        private const double HBar = 1.054571817e-34;
        private const double ElectronCharge = 1.602176634e-19;

        public AdsCft3(double centralCharge, double adsRadius, double fermiVelocity = 5e5)
        {
            CentralCharge = centralCharge;
            AdsRadius = adsRadius;
            FermiVelocity = fermiVelocity;
            NewtonConstant3D = 3.0 * adsRadius / (2.0 * centralCharge);
        }

        public double BulkToBoundaryPropagator(double z, double x, double xPrime, double delta)
        {
            double rSq = z * z + (x - xPrime) * (x - xPrime);
            return Math.Pow(z / rSq, delta);
        }

        public double ScalingDimensionFromMass(double massSqTimesLSq)
        {
            double d = 1.0;
            double discriminant = d * d / 4.0 + massSqTimesLSq;
            if (discriminant < 0) throw new ArgumentException("Below Breitenlohner-Freedman bound");
            return d / 2.0 + Math.Sqrt(discriminant);
        }

        public double TwoPointFunction(double x, double xPrime, double delta)
        {
            double sep = Math.Abs(x - xPrime);
            if (sep < 1e-15) throw new ArgumentException("Coincident points");
            double C = Math.Pow(2.0, 2 * delta - 1);
            return C / Math.Pow(sep, 2 * delta);
        }

        public double OpticalConductivityDC()
        {
            double eSqOverH = 3.87404e-5;
            return eSqOverH * CentralCharge / 2.0;
        }

        public double WiedemannFranzRatio()
        {
            double L0 = Math.PI * Math.PI / 3.0 * KBoltzmann * KBoltzmann / (ElectronCharge * ElectronCharge);
            return L0 * (1.0 - 3.0 / CentralCharge);
        }

        public double LyapunovExponent(double temperature)
        {
            double bound = 2.0 * Math.PI * KBoltzmann * temperature / HBar;
            return bound * (1.0 - 6.0 / (CentralCharge * CentralCharge));
        }

        public double EntanglementEntropy(double intervalLength, double uvCutoff)
        {
            if (intervalLength <= uvCutoff) throw new ArgumentException("Interval must exceed UV cutoff");
            return (CentralCharge / 3.0) * Math.Log(intervalLength / uvCutoff);
        }

        public double HolographicCoherenceEnhancement(double systemSize, double coherenceLength)
        {
            double ratio = systemSize / coherenceLength;
            if (ratio <= 0) throw new ArgumentException("systemSize must exceed coherenceLength");
            return Math.Pow(ratio, CentralCharge / 6.0);
        }

        public double ScramblingTime(double temperature, int nQubits)
        {
            double beta = HBar / (KBoltzmann * temperature);
            return (beta / (2.0 * Math.PI)) * Math.Log(nQubits);
        }

        public double BtzHawkingTemperature(double mass)
        {
            double rPlus = AdsRadius * Math.Sqrt(mass);
            return HBar * rPlus / (2.0 * Math.PI * KBoltzmann * AdsRadius * AdsRadius);
        }
    }
}