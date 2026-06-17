from __future__ import annotations

import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type
from torch.utils.data import DataLoader, TensorDataset
from holographic_qc.ml.pytorch_models import HolographicDecoherenceNet, HolographicTrainer


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class LearningRateScheduler:
    def __init__(self, optimizer, factor: float = 0.5, patience: int = 10, min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.counter = 0
        self.best_loss = float("inf")

    def step(self, val_loss: float):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                for pg in self.optimizer.param_groups:
                    pg["lr"] = max(pg["lr"] * self.factor, self.min_lr)
                self.counter = 0

    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


class TrainingPipeline:
    def __init__(
        self,
        model_class: Type[nn.Module],
        model_kwargs: dict,
        optimizer_class: Type[optim.Optimizer] = optim.Adam,
        optimizer_kwargs: dict = None,
        device: str = "cpu",
        output_dir: str = "models/",
        experiment_name: str = "holographic_qc",
    ):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-3, "weight_decay": 1e-5}
        self.device = device
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.history: Dict[str, List[float]] = {}

    def build_model(self) -> nn.Module:
        self.model = self.model_class(**self.model_kwargs).to(self.device)
        self.optimizer = self.optimizer_class(
            self.model.parameters(), **self.optimizer_kwargs
        )
        return self.model

    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        y_rate: torch.Tensor,
        y_enh: torch.Tensor,
    ) -> torch.Tensor:
        import torch.nn.functional as F
        loss_rate = F.mse_loss(
            torch.log(outputs["decoherence_rate"] + 1e-10),
            torch.log(y_rate + 1e-10),
        )
        loss_enh = F.l1_loss(outputs["enhancement_factor"], y_enh)
        return loss_rate + 0.5 * loss_enh

    def _train_epoch(self, loader: DataLoader, early_stop: EarlyStopping) -> float:
        self.model.train()
        total = 0.0
        n = 0
        for X, y_rate, y_enh in loader:
            X = X.to(self.device)
            y_rate = y_rate.to(self.device)
            y_enh = y_enh.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self._compute_loss(outputs, y_rate, y_enh)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item()
            n += 1
        return total / max(n, 1)

    def _val_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for X, y_rate, y_enh in loader:
                X = X.to(self.device)
                y_rate = y_rate.to(self.device)
                y_enh = y_enh.to(self.device)
                outputs = self.model(X)
                loss = self._compute_loss(outputs, y_rate, y_enh)
                total += loss.item()
                n += 1
        return total / max(n, 1)

    def train(
        self,
        X_train: np.ndarray,
        y_rate_train: np.ndarray,
        y_enh_train: np.ndarray,
        X_val: np.ndarray,
        y_rate_val: np.ndarray,
        y_enh_val: np.ndarray,
        epochs: int = 200,
        batch_size: int = 64,
        patience: int = 20,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        if self.model is None:
            self.build_model()

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_rate_train),
            torch.FloatTensor(y_enh_train),
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_rate_val),
            torch.FloatTensor(y_enh_val),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        early_stop = EarlyStopping(patience=patience)
        lr_scheduler = LearningRateScheduler(self.optimizer)

        train_losses, val_losses, lr_history = [], [], []
        best_val = float("inf")
        best_checkpoint = self.output_dir / f"{self.experiment_name}_best.pt"
        t0 = time.perf_counter()

        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader, early_stop)
            val_loss = self._val_epoch(val_loader)
            lr = lr_scheduler.current_lr()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            lr_history.append(lr)

            lr_scheduler.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), best_checkpoint)

            if verbose and (epoch % 10 == 0 or epoch < 5):
                elapsed = time.perf_counter() - t0
                print(
                    f"Epoch {epoch:04d} | "
                    f"Train: {train_loss:.6f} | "
                    f"Val: {val_loss:.6f} | "
                    f"LR: {lr:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )

            if early_stop.step(val_loss):
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}")
                break

        self.model.load_state_dict(torch.load(best_checkpoint))
        self.history = {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "learning_rate": lr_history,
        }

        history_path = self.output_dir / f"{self.experiment_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(
                {k: [float(v) for v in vals] for k, vals in self.history.items()},
                f, indent=2,
            )
        return self.history

    def evaluate(
        self,
        X_test: np.ndarray,
        y_rate_test: np.ndarray,
        y_enh_test: np.ndarray,
    ) -> dict:
        import torch.nn.functional as F
        self.model.eval()
        X_t = torch.FloatTensor(X_test).to(self.device)
        y_rate_t = torch.FloatTensor(y_rate_test).to(self.device)
        y_enh_t = torch.FloatTensor(y_enh_test).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_t)
            mse_rate = F.mse_loss(outputs["decoherence_rate"], y_rate_t).item()
            mae_rate = F.l1_loss(outputs["decoherence_rate"], y_rate_t).item()
            mse_enh = F.mse_loss(outputs["enhancement_factor"], y_enh_t).item()
            mae_enh = F.l1_loss(outputs["enhancement_factor"], y_enh_t).item()
            loss = self._compute_loss(outputs, y_rate_t, y_enh_t).item()
        results = {
            "total_loss": loss,
            "mse_decoherence_rate": mse_rate,
            "mae_decoherence_rate": mae_rate,
            "mse_enhancement_factor": mse_enh,
            "mae_enhancement_factor": mae_enh,
        }
        results_path = self.output_dir / f"{self.experiment_name}_test_results.json"
        with open(results_path, "w") as f:
            json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
        return results

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_t)
        return {
            "decoherence_rate": outputs["decoherence_rate"].cpu().numpy(),
            "enhancement_factor": outputs["enhancement_factor"].cpu().numpy(),
        }

    def save(self, path: Optional[str] = None):
        save_path = path or str(self.output_dir / f"{self.experiment_name}_final.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_kwargs": self.model_kwargs,
                "history": self.history,
            },
            save_path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        if self.model is None:
            self.build_model()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint.get("history", {})


class CrossValidationRunner:
    def __init__(
        self,
        model_class: Type[nn.Module],
        model_kwargs: dict,
        n_folds: int = 5,
        device: str = "cpu",
    ):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.n_folds = n_folds
        self.device = device
        self.fold_results: List[dict] = []

    def run(
        self,
        X: np.ndarray,
        y_rate: np.ndarray,
        y_enh: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
    ) -> dict:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\nFold {fold_idx + 1}/{self.n_folds}")
            X_tr, X_v = X[train_idx], X[val_idx]
            yr_tr, yr_v = y_rate[train_idx], y_rate[val_idx]
            ye_tr, ye_v = y_enh[train_idx], y_enh[val_idx]

            pipeline = TrainingPipeline(
                model_class=self.model_class,
                model_kwargs=self.model_kwargs,
                device=self.device,
                output_dir=f"models/cv_fold_{fold_idx}/",
                experiment_name=f"fold_{fold_idx}",
            )
            pipeline.build_model()
            pipeline.train(X_tr, yr_tr, ye_tr, X_v, yr_v, ye_v,
                          epochs=epochs, batch_size=batch_size, verbose=False)
            metrics = pipeline.evaluate(X_v, yr_v, ye_v)
            fold_metrics.append(metrics)
            self.fold_results.append(metrics)
            print(f"  Val loss: {metrics['total_loss']:.6f}")

        summary = {}
        for key in fold_metrics[0].keys():
            vals = [m[key] for m in fold_metrics]
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals))

        print("\nCross-validation summary:")
        for k, v in summary.items():
            print(f"  {k}: {v:.6e}")
        return summary