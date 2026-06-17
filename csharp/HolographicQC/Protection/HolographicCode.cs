using System;
using System.Collections.Generic;
using System.Linq;
using HolographicQC.Core;

namespace HolographicQC.Protection
{
    public class SurfaceCode
    {
        public int Distance { get; }
        public int N { get; }
        public int K { get; } = 1;

        public SurfaceCode(int distance)
        {
            Distance = distance;
            N = distance * distance + (distance - 1) * (distance - 1);
        }

        public double HolographicCodeDistance(double centralCharge, double systemSize, double latticeSpacing)
        {
            double correction = (centralCharge / (6.0 * Math.PI)) * Math.Log(systemSize / latticeSpacing);
            return Distance * (1.0 + correction);
        }

        public double HolographicThreshold(double centralCharge, double pThreshStd = 0.01)
            => pThreshStd * (1.0 + centralCharge / (12.0 * Math.PI));

        public double LogicalErrorRate(double physicalRate)
        {
            double threshold = 0.01;
            if (physicalRate >= threshold) return 0.5;
            double ratio = physicalRate / threshold;
            return 0.1 * Math.Pow(ratio, Distance / 2);
        }
    }

    public class PentagonHaPPYCode
    {
        public int NLayers { get; }
        public int NBoundary { get; }
        public int NBulk { get; }

        public PentagonHaPPYCode(int nLayers = 3)
        {
            NLayers = nLayers;
            NBoundary = 5 * (int)Math.Pow(4, nLayers - 1);
            NBulk = (5 * ((int)Math.Pow(4, nLayers) - 1)) / 3;
        }

        public double EncodingRate() => (double)NBulk / NBoundary;

        public double ErasureCorrectionRadius(int bulkRegionSize)
            => (double)bulkRegionSize / NBoundary;
    }

    public class HolographicCode
    {
        private readonly string _codeType;
        private readonly double _c;
        private readonly SurfaceCode _surfaceCode;
        private readonly PentagonHaPPYCode _pentagonCode;

        public HolographicCode(string codeType = "pentagon", double centralCharge = 1.0)
        {
            _codeType = codeType;
            _c = centralCharge;
            if (codeType == "surface")
                _surfaceCode = new SurfaceCode(7);
            else
                _pentagonCode = new PentagonHaPPYCode();
        }

        public double EncodingRate()
        {
            if (_codeType == "surface") return 1.0 / (_surfaceCode.N);
            return _pentagonCode.EncodingRate();
        }

        public double EffectiveDistance(double systemSize, double latticeSpacing)
        {
            if (_codeType == "surface")
                return _surfaceCode.HolographicCodeDistance(_c, systemSize, latticeSpacing);
            return _c * Math.Log(systemSize / latticeSpacing);
        }

        public double ErrorThreshold(double pStd = 0.01)
        {
            if (_codeType == "surface")
                return _surfaceCode.HolographicThreshold(_c, pStd);
            return pStd * (1.0 + _c / (12.0 * Math.PI));
        }

        public double LogicalErrorRate(double physicalRate, double systemSize, double latticeSpacing)
        {
            double pThresh = ErrorThreshold();
            if (physicalRate >= pThresh) return 0.5;
            double dEff = EffectiveDistance(systemSize, latticeSpacing);
            double ratio = physicalRate / pThresh;
            return 0.1 * Math.Pow(ratio, (int)dEff / 2);
        }

        public Dictionary<string, object> ResourceOverhead(int nLogical)
        {
            double rate = EncodingRate();
            int nPhys = (int)Math.Ceiling(nLogical / Math.Max(rate, 1e-10));
            return new Dictionary<string, object>
            {
                ["n_logical_qubits"] = nLogical,
                ["n_physical_qubits"] = nPhys,
                ["overhead_ratio"] = nPhys / (double)nLogical,
                ["encoding_rate"] = rate,
            };
        }
    }
}