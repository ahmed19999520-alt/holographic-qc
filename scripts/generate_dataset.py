import argparse
from holographic_qc.utils.datasets import SyntheticDatasetGenerator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--noise_level", type=float, default=0.05)
    return parser.parse_args()

def main():
    args = parse_args()
    gen = SyntheticDatasetGenerator(seed=args.seed)
    datasets = gen.save_all(data_dir=args.output_dir)
    for name, df in datasets.items():
        print(f"{name}: {df.shape[0]} samples x {df.shape[1]} features")
        print(df.describe().to_string())
        print()

if __name__ == "__main__":
    main()