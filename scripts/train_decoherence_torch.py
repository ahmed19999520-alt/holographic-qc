import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pathlib import Path

from holographic_qc.ml.pytorch_models import HolographicDecoherenceNet, HolographicTrainer
from holographic_qc.utils.datasets import RealDataLoader
from holographic_qc.materials.bi2se3 import Bi2Se3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--data", type=str, default="data/bi2se3_arpes.csv")
    parser.add_argument("--output", type=str, default="models/decoherence_torch/")
    parser.add_argument("--central_charge", type=float, default=1.0)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 256, 128, 64])
    parser.add_argument("--device", type=str, default="cpu")
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
    target_cols = ["t2_standard_ns", "enhancement_factor"]

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = loader.prepare_training_features(
        df, feature_cols, target_cols
    )

    y_rate_train = y_train[:, 0:1]
    y_enh_train = y_train[:, 1:2]
    y_rate_val = y_val[:, 0:1]
    y_enh_val = y_val[:, 1:2]
    y_rate_test = y_test[:, 0:1]
    y_enh_test = y_test[:, 1:2]

    model = HolographicDecoherenceNet(
        input_dim=X_train.shape[1],
        central_charge=args.central_charge,
        hidden_dims=args.hidden_dims,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    trainer = HolographicTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
    )

    history = trainer.fit(
        X_train, y_rate_train, y_enh_train,
        X_val, y_rate_val, y_enh_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    with open(f"{args.output}/training_history.json", "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2
        )

    torch.save(model.state_dict(), f"{args.output}/final_model.pt")
    torch.save({"model_config": {
        "input_dim": X_train.shape[1],
        "central_charge": args.central_charge,
        "hidden_dims": args.hidden_dims,
    }}, f"{args.output}/model_config.pt")

    model.eval()
    mat = Bi2Se3()
    test_input = scaler.transform(
        np.array([[4.0, 1e-6, mat.fermi_velocity, mat.xi * 1e9, mat.central_charge, 1e-6 / mat.xi]])
    ).astype(np.float32)
    with torch.no_grad():
        pred = model(torch.FloatTensor(test_input))
    print(f"\nSample prediction at T=4K, L=1um:")
    print(f"  Decoherence rate: {pred['decoherence_rate'].item():.6e}")
    print(f"  Enhancement factor: {pred['enhancement_factor'].item():.4f}")

    model.eval()
    X_test_t = torch.FloatTensor(X_test)
    y_rate_t = torch.FloatTensor(y_rate_test)
    y_enh_t = torch.FloatTensor(y_enh_test)
    with torch.no_grad():
        pred_test = model(X_test_t)
        import torch.nn.functional as F
        mse_rate = F.mse_loss(pred_test["decoherence_rate"], y_rate_t).item()
        mse_enh = F.mse_loss(pred_test["enhancement_factor"], y_enh_t).item()
    print(f"\nTest MSE decoherence rate: {mse_rate:.6e}")
    print(f"Test MSE enhancement factor: {mse_enh:.6e}")
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()