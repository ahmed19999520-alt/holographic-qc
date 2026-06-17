from __future__ import annotations

import numpy as np
import scipy.optimize as opt
from typing import Callable, List, Optional, Tuple


class PauliOperator:
    SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)

    @staticmethod
    def kron_op(op: np.ndarray, site: int, n_sites: int) -> np.ndarray:
        result = np.eye(1, dtype=np.complex128)
        for s in range(n_sites):
            result = np.kron(result, op if s == site else PauliOperator.I2)
        return result

    @staticmethod
    def xx_coupling(i: int, j: int, n: int) -> np.ndarray:
        Xi = PauliOperator.kron_op(PauliOperator.SIGMA_X, i, n)
        Xj = PauliOperator.kron_op(PauliOperator.SIGMA_X, j, n)
        return Xi @ Xj

    @staticmethod
    def zz_coupling(i: int, j: int, n: int) -> np.ndarray:
        Zi = PauliOperator.kron_op(PauliOperator.SIGMA_Z, i, n)
        Zj = PauliOperator.kron_op(PauliOperator.SIGMA_Z, j, n)
        return Zi @ Zj

    @staticmethod
    def transverse_field(i: int, n: int) -> np.ndarray:
        return PauliOperator.kron_op(PauliOperator.SIGMA_Z, i, n)


class VariationalAnsatz:
    def __init__(self, n_qubits: int, depth: int):
        self.n = n_qubits
        self.d = depth
        self.n_params = depth * n_qubits * 2

    def initial_params(self, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform(-np.pi, np.pi, self.n_params)

    def state_vector(self, params: np.ndarray) -> np.ndarray:
        dim = 2**self.n
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0
        param_idx = 0
        for layer in range(self.d):
            for qubit in range(self.n):
                theta = params[param_idx]
                phi = params[param_idx + 1]
                param_idx += 2
                state = self._apply_ry(state, qubit, theta)
                state = self._apply_rz(state, qubit, phi)
            for qubit in range(self.n - 1):
                state = self._apply_cnot(state, qubit, qubit + 1)
        return state

    def _apply_ry(self, state: np.ndarray, qubit: int, theta: float) -> np.ndarray:
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        Ry = np.array([[c, -s], [s, c]], dtype=np.complex128)
        U = PauliOperator.kron_op(Ry, qubit, self.n)
        return U @ state

    def _apply_rz(self, state: np.ndarray, qubit: int, phi: float) -> np.ndarray:
        Rz = np.diag([np.exp(-1j * phi / 2), np.exp(1j * phi / 2)]).astype(np.complex128)
        U = PauliOperator.kron_op(Rz, qubit, self.n)
        return U @ state

    def _apply_cnot(self, state: np.ndarray, ctrl: int, tgt: int) -> np.ndarray:
        dim = 2**self.n
        result = state.copy()
        for idx in range(dim):
            ctrl_bit = (idx >> (self.n - 1 - ctrl)) & 1
            if ctrl_bit == 1:
                tgt_bit = (self.n - 1 - tgt)
                flipped = idx ^ (1 << tgt_bit)
                result[idx], result[flipped] = state[flipped], state[idx]
        return result

    def gradient(self, params: np.ndarray, hamiltonian: np.ndarray) -> np.ndarray:
        grad = np.zeros(len(params))
        shift = np.pi / 2.0
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += shift
            params_minus[i] -= shift
            e_plus = self.energy(params_plus, hamiltonian)
            e_minus = self.energy(params_minus, hamiltonian)
            grad[i] = (e_plus - e_minus) / 2.0
        return grad

    def energy(self, params: np.ndarray, hamiltonian: np.ndarray) -> float:
        state = self.state_vector(params)
        return float(np.real(state.conj() @ hamiltonian @ state))


class VQE:
    def __init__(
        self,
        hamiltonian: np.ndarray,
        n_qubits: int,
        depth: int = 3,
        method: str = "BFGS",
        max_iter: int = 500,
        tol: float = 1e-8,
    ):
        self.H = hamiltonian
        self.n = n_qubits
        self.ansatz = VariationalAnsatz(n_qubits, depth)
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.energy_history: List[float] = []

    def _objective(self, params: np.ndarray) -> float:
        e = self.ansatz.energy(params, self.H)
        self.energy_history.append(e)
        return e

    def _gradient(self, params: np.ndarray) -> np.ndarray:
        return self.ansatz.gradient(params, self.H)

    def run(self, seed: int = 42) -> dict:
        params0 = self.ansatz.initial_params(seed)
        self.energy_history = []
        result = opt.minimize(
            self._objective,
            params0,
            jac=self._gradient,
            method=self.method,
            options={"maxiter": self.max_iter, "gtol": self.tol},
        )
        optimal_state = self.ansatz.state_vector(result.x)
        exact_eigvals = np.linalg.eigvalsh(self.H)
        exact_ground = float(exact_eigvals[0])
        return {
            "optimal_energy": result.fun,
            "optimal_params": result.x,
            "optimal_state": optimal_state,
            "exact_ground_energy": exact_ground,
            "energy_error": abs(result.fun - exact_ground),
            "relative_error": abs(result.fun - exact_ground) / abs(exact_ground),
            "n_iterations": result.nit,
            "converged": result.success,
            "energy_history": self.energy_history,
        }

    def energy_landscape_1d(
        self, params: np.ndarray, param_idx: int, n_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        thetas = np.linspace(-np.pi, np.pi, n_points)
        energies = np.zeros(n_points)
        for i, th in enumerate(thetas):
            p = params.copy()
            p[param_idx] = th
            energies[i] = self.ansatz.energy(p, self.H)
        return thetas, energies

    def variance(self, params: np.ndarray) -> float:
        state = self.ansatz.state_vector(params)
        E = float(np.real(state.conj() @ self.H @ state))
        H_sq = self.H @ self.H
        E2 = float(np.real(state.conj() @ H_sq @ state))
        return E2 - E**2

    @staticmethod
    def ising_hamiltonian(n: int, J: float = 1.0, h: float = 1.0) -> np.ndarray:
        dim = 2**n
        H = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(n - 1):
            H -= J * PauliOperator.xx_coupling(i, i + 1, n)
        for i in range(n):
            H -= h * PauliOperator.transverse_field(i, n)
        return H

    @staticmethod
    def heisenberg_hamiltonian(n: int, J: float = 1.0) -> np.ndarray:
        dim = 2**n
        H = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(n - 1):
            H += J * PauliOperator.xx_coupling(i, i + 1, n)
            Xi = PauliOperator.kron_op(PauliOperator.SIGMA_X, i, n)
            Yi = PauliOperator.kron_op(PauliOperator.SIGMA_Y, i, n)
            Xi1 = PauliOperator.kron_op(PauliOperator.SIGMA_X, i + 1, n)
            Yi1 = PauliOperator.kron_op(PauliOperator.SIGMA_Y, i + 1, n)
            H += J * (Xi @ Xi1 + Yi @ Yi1 + PauliOperator.zz_coupling(i, i + 1, n))
        return H