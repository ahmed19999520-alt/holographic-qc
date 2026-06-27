from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, TensorDataset
from collections import deque


@dataclass
class ContinualConfig:
    ewc_lambda: float = 1000.0
    replay_buffer_size: int = 2000
    replay_batch_size: int = 64
    progressive_n_columns: int = 3
    learning_rate: float = 1e-3
    n_epochs_per_task: int = 10


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)

    def push(self, x: np.ndarray, y: np.ndarray, task_id: int = 0):
        self._buffer.append((x.copy(), y.copy(), task_id))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self._buffer) == 0:
            raise RuntimeError("Empty buffer")
        indices = np.random.choice(len(self._buffer), min(batch_size, len(self._buffer)), replace=False)
        xs, ys, tasks = [], [], []
        for i in indices:
            x, y, t = self._buffer[i]
            xs.append(x); ys.append(y); tasks.append(t)
        return np.stack(xs), np.stack(ys), np.array(tasks)

    def __len__(self) -> int:
        return len(self._buffer)


class EWC:
    def __init__(self, model: nn.Module, config: ContinualConfig, device: str = "cpu"):
        self.model = model
        self.cfg = config
        self.device = device
        self._fisher: Dict[str, torch.Tensor] = {}
        self._optima: Dict[str, torch.Tensor] = {}
        self._n_tasks_seen: int = 0

    def compute_fisher(self, dataloader: DataLoader, n_samples: int = 200) -> Dict[str, torch.Tensor]:
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self.model.eval()
        n_seen = 0
        for X, y in dataloader:
            X = X.to(self.device)
            y = y.to(self.device)
            self.model.zero_grad()
            output = self.model(X)
            log_probs = F.log_softmax(output, dim=-1) if output.dim() > 1 else output
            loss = F.nll_loss(log_probs, y.long()) if output.dim() > 1 else F.mse_loss(output, y)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)
            n_seen += len(X)
            if n_seen >= n_samples:
                break
        for name in fisher:
            fisher[name] /= max(n_seen, 1)
        return fisher

    def consolidate(self, dataloader: DataLoader):
        new_fisher = self.compute_fisher(dataloader)
        for name, param in self.model.named_parameters():
            if name in self._fisher:
                self._fisher[name] = (self._fisher[name] * self._n_tasks_seen + new_fisher[name]) / (self._n_tasks_seen + 1)
            else:
                self._fisher[name] = new_fisher[name].clone()
            self._optima[name] = param.data.clone()
        self._n_tasks_seen += 1

    def penalty(self) -> torch.Tensor:
        if not self._fisher:
            return torch.tensor(0.0, device=self.device)
        loss = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            if name in self._fisher and name in self._optima:
                diff = param - self._optima[name]
                loss += (self._fisher[name] * diff.pow(2)).sum()
        return (self.cfg.ewc_lambda / 2.0) * loss

    def train_task(
        self, dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        n_epochs: int = None,
    ) -> List[float]:
        n_epochs = n_epochs or self.cfg.n_epochs_per_task
        losses = []
        self.model.train()
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for X, y in dataloader:
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                output = self.model(X)
                task_loss = loss_fn(output, y)
                ewc_loss = self.penalty()
                total_loss = task_loss + ewc_loss
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += total_loss.item()
                n_batches += 1
            losses.append(epoch_loss / max(n_batches, 1))
        return losses


class ProgressiveNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_columns: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_columns = n_columns
        self._columns: nn.ModuleList = nn.ModuleList()
        self._adapters: nn.ModuleList = nn.ModuleList()
        self._active_column: int = -1
        self._add_column()

    def _add_column(self):
        col_idx = len(self._columns)
        lateral_in = self.hidden_dim * col_idx
        column = nn.Sequential(
            nn.Linear(self.input_dim + lateral_in, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        self._columns.append(column)
        self._active_column = col_idx

    def add_task(self):
        if len(self._columns) < self.n_columns:
            for col in self._columns[:-1]:
                for param in col.parameters():
                    param.requires_grad_(False)
            self._add_column()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prev_outputs = []
        for i, col in enumerate(self._columns):
            if i == self._active_column:
                if prev_outputs:
                    lateral = torch.cat(prev_outputs, dim=-1)
                    inp = torch.cat([x, lateral], dim=-1)
                else:
                    inp = x
                    if inp.shape[-1] < col[0].in_features:
                        pad = torch.zeros(*inp.shape[:-1], col[0].in_features - inp.shape[-1])
                        inp = torch.cat([inp, pad], dim=-1)
                out = col(inp)
                prev_outputs.append(out)
            elif i < self._active_column:
                with torch.no_grad():
                    if prev_outputs:
                        lateral = torch.cat(prev_outputs, dim=-1)
                        inp = torch.cat([x, lateral], dim=-1)
                    else:
                        inp = x
                        if inp.shape[-1] < col[0].in_features:
                            pad = torch.zeros(*inp.shape[:-1], col[0].in_features - inp.shape[-1])
                            inp = torch.cat([inp, pad], dim=-1)
                    out = col(inp)
                    prev_outputs.append(out)
        return prev_outputs[self._active_column]


class ExperienceReplay:
    def __init__(self, buffer: ReplayBuffer, config: ContinualConfig, device: str = "cpu"):
        self.buffer = buffer
        self.cfg = config
        self.device = device

    def train_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        current_X: torch.Tensor,
        current_y: torch.Tensor,
        loss_fn,
    ) -> float:
        optimizer.zero_grad()
        current_output = model(current_X.to(self.device))
        current_loss = loss_fn(current_output, current_y.to(self.device))
        replay_loss = torch.tensor(0.0, device=self.device)
        if len(self.buffer) >= self.cfg.replay_batch_size:
            rx, ry, _ = self.buffer.sample(self.cfg.replay_batch_size)
            rx_t = torch.FloatTensor(rx).to(self.device)
            ry_t = torch.FloatTensor(ry).to(self.device)
            replay_output = model(rx_t)
            replay_loss = loss_fn(replay_output, ry_t)
        total = current_loss + 0.5 * replay_loss
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return total.item()

    def fill_buffer(self, X: np.ndarray, y: np.ndarray, task_id: int = 0):
        n = min(len(X), self.cfg.replay_buffer_size // 4)
        indices = np.random.choice(len(X), n, replace=False)
        for i in indices:
            self.buffer.push(X[i], y[i], task_id)


class ContinualLearner:
    def __init__(
        self,
        model: nn.Module,
        config: ContinualConfig,
        strategy: str = "ewc",
        device: str = "cpu",
    ):
        self.model = model
        self.cfg = config
        self.strategy = strategy
        self.device = device
        self._n_tasks = 0
        self._task_performance: List[Dict] = []
        self._replay_buffer = ReplayBuffer(config.replay_buffer_size)

        if strategy == "ewc":
            self.ewc = EWC(model, config, device)
        elif strategy == "replay":
            self.replay = ExperienceReplay(self._replay_buffer, config, device)
        elif strategy == "progressive":
            pass

    def learn_task(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        task_id: Optional[int] = None,
    ) -> dict:
        task_id = task_id if task_id is not None else self._n_tasks
        self._n_tasks += 1

        X_t = torch.FloatTensor(X_train)
        y_t = torch.FloatTensor(y_train)
        X_v = torch.FloatTensor(X_val)
        y_v = torch.FloatTensor(y_val)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)

        def loss_fn(output, target):
            if output.shape == target.shape:
                return F.mse_loss(output, target)
            return F.cross_entropy(output, target.long())

        train_losses = []
        if self.strategy == "ewc":
            train_losses = self.ewc.train_task(loader, optimizer, loss_fn, self.cfg.n_epochs_per_task)
            self.ewc.consolidate(loader)
        elif self.strategy == "replay":
            for epoch in range(self.cfg.n_epochs_per_task):
                ep_loss = 0.0
                n_b = 0
                for X_b, y_b in loader:
                    loss = self.replay.train_step(self.model, optimizer, X_b, y_b, loss_fn)
                    ep_loss += loss
                    n_b += 1
                train_losses.append(ep_loss / max(n_b, 1))
            self.replay.fill_buffer(X_train, y_train, task_id)

        self.model.eval()
        with torch.no_grad():
            val_out = self.model(X_v.to(self.device))
            val_loss = loss_fn(val_out, y_v.to(self.device)).item()

        result = {
            "task_id": task_id,
            "strategy": self.strategy,
            "final_train_loss": train_losses[-1] if train_losses else 0.0,
            "val_loss": val_loss,
            "n_epochs": self.cfg.n_epochs_per_task,
        }
        self._task_performance.append(result)
        return result

    def backward_transfer(self) -> float:
        if len(self._task_performance) < 2:
            return 0.0
        losses = [p["val_loss"] for p in self._task_performance]
        return float(np.mean(np.diff(losses)))

    def plasticity(self) -> float:
        if not self._task_performance:
            return 0.0
        return float(self._task_performance[-1]["val_loss"])

    def statistics(self) -> dict:
        return {
            "n_tasks_learned": self._n_tasks,
            "strategy": self.strategy,
            "backward_transfer": self.backward_transfer(),
            "plasticity": self.plasticity(),
            "task_history": self._task_performance,
        }