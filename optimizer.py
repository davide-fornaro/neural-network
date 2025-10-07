import numpy as np
from layer import Layer


class Optimizer:
    """Base optimizer class that operates on layers with trainable parameters."""
    def __init__(self, layers: list[Layer]):
        self.layers = [L for L in layers if hasattr(L, "W") and hasattr(L, "b")]

    def step(self) -> None:
        """Perform a single optimization step (update parameters)."""
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent optimizer."""
    def __init__(self, layers, lr: float = 1e-1) -> None:
        super().__init__(layers)
        self.lr = np.float32(lr)

    def step(self) -> None:
        for L in self.layers:
            L.W -= self.lr * L.dW  # type: ignore[attr-defined]
            L.b -= self.lr * L.db  # type: ignore[attr-defined]


class SGDMomentum(Optimizer):
    """SGD with momentum."""
    def __init__(self, layers, lr: float = 1e-2, beta: float = 0.9) -> None:
        super().__init__(layers)
        self.lr = np.float32(lr)
        self.beta = np.float32(beta)
        self.vW = {i: np.zeros_like(L.W) for i, L in enumerate(self.layers)}  # type: ignore[attr-defined]
        self.vb = {i: np.zeros_like(L.b) for i, L in enumerate(self.layers)}  # type: ignore[attr-defined]

    def step(self) -> None:
        one_minus_beta = np.float32(1.0) - self.beta
        for i, L in enumerate(self.layers):
            self.vW[i] = self.beta * self.vW[i] + one_minus_beta * L.dW  # type: ignore[attr-defined]
            self.vb[i] = self.beta * self.vb[i] + one_minus_beta * L.db  # type: ignore[attr-defined]
            L.W -= self.lr * self.vW[i]  # type: ignore[attr-defined]
            L.b -= self.lr * self.vb[i]  # type: ignore[attr-defined]


class Adam(Optimizer):
    """Adam optimizer."""
    def __init__(self, layers, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        super().__init__(layers)
        self.lr = np.float32(lr)
        self.beta1 = np.float32(beta1)
        self.beta2 = np.float32(beta2)
        self.eps = np.float32(eps)
        self.t = 0
        self.mW = {i: np.zeros_like(L.W) for i, L in enumerate(self.layers)}  # type: ignore[attr-defined]
        self.vW = {i: np.zeros_like(L.W) for i, L in enumerate(self.layers)}  # type: ignore[attr-defined]
        self.mb = {i: np.zeros_like(L.b) for i, L in enumerate(self.layers)}  # type: ignore[attr-defined]
        self.vb = {i: np.zeros_like(L.b) for i, L in enumerate(self.layers)}  # type: ignore[attr-defined]

    def step(self) -> None:
        self.t += 1
        beta1_pow = np.float32(1.0) - self.beta1 ** self.t
        beta2_pow = np.float32(1.0) - self.beta2 ** self.t
        for i, L in enumerate(self.layers):
            self.mW[i] = self.beta1 * self.mW[i] + (np.float32(1.0) - self.beta1) * L.dW  # type: ignore[attr-defined]
            self.vW[i] = self.beta2 * self.vW[i] + (np.float32(1.0) - self.beta2) * (L.dW ** 2)  # type: ignore[attr-defined]
            self.mb[i] = self.beta1 * self.mb[i] + (np.float32(1.0) - self.beta1) * L.db  # type: ignore[attr-defined]
            self.vb[i] = self.beta2 * self.vb[i] + (np.float32(1.0) - self.beta2) * (L.db ** 2)  # type: ignore[attr-defined]
            mW_hat = self.mW[i] / beta1_pow
            vW_hat = self.vW[i] / beta2_pow
            mb_hat = self.mb[i] / beta1_pow
            vb_hat = self.vb[i] / beta2_pow
            L.W -= self.lr * mW_hat / (np.sqrt(vW_hat, dtype=np.float32) + self.eps)  # type: ignore[attr-defined]
            L.b -= self.lr * mb_hat / (np.sqrt(vb_hat, dtype=np.float32) + self.eps)  # type: ignore[attr-defined]


class AdamW(Optimizer):
    """Adam optimizer with weight decay."""
    def __init__(
        self,
        layers,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__(layers)
        self.lr = np.float32(lr)
        self.beta1 = np.float32(beta1)
        self.beta2 = np.float32(beta2)
        self.eps = np.float32(eps)
        self.weight_decay = np.float32(weight_decay)
        self.t = 0
        self.mW = {i: np.zeros_like(L.W) for i, L in enumerate(self.layers)}  # type: ignore[attr-defined]
        self.vW = {i: np.zeros_like(L.W) for i, L in enumerate(self.layers)}  # type: ignore[attr-defined]
        self.mb = {i: np.zeros_like(L.b) for i, L in enumerate(self.layers)}  # type: ignore[attr-defined]
        self.vb = {i: np.zeros_like(L.b) for i, L in enumerate(self.layers)}  # type: ignore[attr-defined]

    def step(self) -> None:
        self.t += 1
        beta1_pow = np.float32(1.0) - self.beta1 ** self.t
        beta2_pow = np.float32(1.0) - self.beta2 ** self.t
        for i, L in enumerate(self.layers):
            self.mW[i] = self.beta1 * self.mW[i] + (np.float32(1.0) - self.beta1) * L.dW  # type: ignore[attr-defined]
            self.vW[i] = self.beta2 * self.vW[i] + (np.float32(1.0) - self.beta2) * (L.dW ** 2)  # type: ignore[attr-defined]
            self.mb[i] = self.beta1 * self.mb[i] + (np.float32(1.0) - self.beta1) * L.db  # type: ignore[attr-defined]
            self.vb[i] = self.beta2 * self.vb[i] + (np.float32(1.0) - self.beta2) * (L.db ** 2)  # type: ignore[attr-defined]
            mW_hat = self.mW[i] / beta1_pow
            vW_hat = self.vW[i] / beta2_pow
            mb_hat = self.mb[i] / beta1_pow
            vb_hat = self.vb[i] / beta2_pow
            weight_decay_term = self.weight_decay * L.W  # type: ignore[attr-defined]
            L.W -= self.lr * (mW_hat / (np.sqrt(vW_hat, dtype=np.float32) + self.eps) + weight_decay_term)  # type: ignore[attr-defined]
            L.b -= self.lr * mb_hat / (np.sqrt(vb_hat, dtype=np.float32) + self.eps)  # type: ignore[attr-defined]
