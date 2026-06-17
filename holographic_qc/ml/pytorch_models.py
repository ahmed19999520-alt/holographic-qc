from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple


class VirasoroEquivariantLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, central_charge: float):
        super().__init__()
        self.c = central_charge
        self.linear = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.tensor([central_charge / 6.0]), requires_grad=False)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * self.scale


class BulkPropagatorLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, ads_radius: float, delta: float):
        super().__init__()
        self.L = ads_radius
        self.delta = delta
        self.net = nn.Sequential(
            nn.Linear(in_features + 2, out_features),
            nn.GELU(),
            nn.Linear(out_features, out_features),
        )
        self.log_z = nn.Parameter(torch.zeros(1))
        self.log_r = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        K = (z / (z**2 + r**2 + 1e-10))**self.delta
        K_feat = K.unsqueeze(-1).expand_as(x[:, :1].expand(-1, 1)).repeat(1, x.shape[1] // x.shape[1])
        inp = torch.cat([x, z.unsqueeze(-1), r.unsqueeze(-1)], dim=-1)
        return self.net(inp) * K_feat.mean(dim=-1, keepdim=True)


class HolographicDecoherenceNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        central_charge: float,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.c = central_charge
        hidden_dims = hidden_dims or [128, 256, 128, 64]

        layers_list = []
        prev_dim = input_dim
        for h_dim in hidden_dims[:-1]:
            layers_list.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = h_dim
        self.feature_net = nn.Sequential(*layers_list)

        self.virasoro_layer = VirasoroEquivariantLayer(prev_dim, hidden_dims[-1], central_charge)

        self.decoherence_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

        self.enhancement_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.feature_net(x)
        vir = F.gelu(self.virasoro_layer(feat))
        decoherence_rate = self.decoherence_head(vir)
        enhancement = self.enhancement_head(vir)
        return {
            "decoherence_rate": decoherence_rate,
            "enhancement_factor": enhancement,
        }


class EntanglementEntropyNet(nn.Module):
    def __init__(self, central_charge: float):
        super().__init__()
        self.c_init = central_charge
        self.c = nn.Parameter(torch.tensor([float(central_charge)]))
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, log_ratio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        S_pred = self.net(log_ratio)
        S_theory = (self.c / 3.0) * log_ratio
        return S_pred, S_theory

    def physics_loss(self, log_ratio, S_measured):
        S_pred, S_theory = self(log_ratio)
        data_loss = F.mse_loss(S_pred, S_measured)
        physics_loss = F.mse_loss(S_pred, S_theory)
        return data_loss + 0.1 * physics_loss

    def extract_c(self) -> float:
        return float(self.c.item())


class OTOCNet(nn.Module):
    def __init__(self, n_sites: int, central_charge: float):
        super().__init__()
        self.n = n_sites
        self.c = central_charge
        self.lambda_L = nn.Parameter(torch.tensor([1.0]))

        self.time_enc = nn.Sequential(
            nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 64), nn.GELU()
        )
        self.site_enc = nn.Sequential(
            nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 64), nn.GELU()
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t_feat = self.time_enc(t)
        x_feat = self.site_enc(x)
        combined = torch.cat([t_feat, x_feat], dim=-1)
        F_pred = self.head(combined)
        F_theory = 1.0 - torch.exp(-torch.abs(self.lambda_L) * t)
        return F_pred, F_theory

    def chaos_penalty(self, temperature: float) -> torch.Tensor:
        kB = 1.380649e-23
        hbar = 1.054571817e-34
        bound = 2.0 * math.pi * kB * temperature / hbar
        violation = F.relu(torch.abs(self.lambda_L) - bound)
        return violation**2


class HolographicTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = scheduler
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def train_epoch(self, loader: DataLoader, loss_fn) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            X = batch[0].to(self.device)
            y_rate = batch[1].to(self.device)
            y_enh = batch[2].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = (
                F.mse_loss(torch.log(outputs["decoherence_rate"] + 1e-10),
                           torch.log(y_rate + 1e-10))
                + F.l1_loss(outputs["enhancement_factor"], y_enh)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def val_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in loader:
                X = batch[0].to(self.device)
                y_rate = batch[1].to(self.device)
                y_enh = batch[2].to(self.device)
                outputs = self.model(X)
                loss = (
                    F.mse_loss(torch.log(outputs["decoherence_rate"] + 1e-10),
                               torch.log(y_rate + 1e-10))
                    + F.l1_loss(outputs["enhancement_factor"], y_enh)
                )
                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(n_batches, 1)

    def fit(
        self, X_train: np.ndarray, y_rate_train: np.ndarray, y_enh_train: np.ndarray,
        X_val: np.ndarray, y_rate_val: np.ndarray, y_enh_val: np.ndarray,
        epochs: int = 100, batch_size: int = 32,
    ) -> Dict[str, List[float]]:
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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float("inf")
        patience_counter = 0
        patience = 20

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, None)
            val_loss = self.val_epoch(val_loader)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_model.pt")
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(f"Epoch {epoch:04d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        self.model.load_state_dict(torch.load("best_model.pt"))
        return {"train_loss": self.train_losses, "val_loss": self.val_losses}