import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from holographic_qc.ml.tensorflow_models import HolographicDecoherenceNet, HolographicTrainer, build_decoherence_model
from holographic_qc.utils.datasets import RealDataLoader
from holographic_qc.materials.bi2se3 import Bi2Se3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--data", type=str, default="data/bi2se3_arpes.csv")
    parser.add_argument("--output", type=str, default="models/decoherence_tf/")
    parser.add_argument("--central_charge", type=float, default=1.0)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 256, 128, 64])
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)

    loader = RealDataLoader("data")
    df = loader.load_bi2se3()

    feature_cols = [
        "temperature_K", "system_size_m", "fermi_velocity",
        "coherence_length_nm", "central_charge", "ratio_L_xi"
    ]
    target_rate_col = ["t2_standard_ns"]
    target_enh_col = ["enhancement_factor"]

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = loader.prepare_training_features(
        df, feature_cols, target_rate_col + target_enh_col
    )

    y_rate_train = y_train[:, 0:1]
    y_enh_train = y_train[:, 1:2]
    y_rate_val = y_val[:, 0:1]
    y_enh_val = y_val[:, 1:2]
    y_rate_test = y_test[:, 0:1]
    y_enh_test = y_test[:, 1:2]

    model = build_decoherence_model(
        input_dim=X_train.shape[1],
        central_charge=args.central_charge,
        hidden_dims=args.hidden_dims,
        learning_rate=args.learning_rate,
    )

    trainer = HolographicTrainer(
        model=model,
        config={"checkpoint_path": f"{args.output}/best.keras"},
    )

    history = trainer.train(
        X_train,
        {"decoherence_rate": y_rate_train, "enhancement_factor": y_enh_train},
        X_val,
        {"decoherence_rate": y_rate_val, "enhancement_factor": y_enh_val},
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    test_results = trainer.evaluate(
        X_test,
        {"decoherence_rate": y_rate_test, "enhancement_factor": y_enh_test},
    )

    with open(f"{args.output}/training_history.json", "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2
        )

    with open(f"{args.output}/test_results.json", "w") as f:
        json.dump({k: float(v) for k, v in test_results.items()}, f, indent=2)

    model.save(f"{args.output}/final_model.keras")

    mat = Bi2Se3()
    test_input = np.array([[4.0, 1e-6, mat.fermi_velocity, mat.xi * 1e9, mat.central_charge, 1e-6 / mat.xi]])
    pred = model(tf.cast(scaler.transform(test_input), tf.float32), training=False)
    print(f"\nSample prediction:")
    print(f"  Decoherence rate: {pred['decoherence_rate'].numpy()[0, 0]:.6e}")
    print(f"  Enhancement factor: {pred['enhancement_factor'].numpy()[0, 0]:.4f}")
    print(f"\nTest metrics: {test_results}")
    print(f"Model and history saved to {args.output}")


if __name__ == "__main__":
    main()