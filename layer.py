import numpy as np
from utils import ReLU, ReLU_deriv, softmax


# X: input data, each column is an example. Dimension: (n_x, m)
# W: weight matrix of a layer. Dimension: (n_out, n_in)
# b: bias vector of a layer, replicated across columns. Dimension: (n_out, 1)
# Z: linear output of the layer (pre-activation). Dimension: (n_out, m)
# A: output after activation. Dimension: (n_out, m)
# Y: target (labels). In one-hot version: (n_y, m)
# dZ: gradient of the loss with respect to Z. Dimension: (n_out, m)
# dA: gradient of the loss with respect to A. Dimension: (n_out, m)
# dW: gradient of the loss with respect to W. Dimension: (n_out, n_in)
# db: gradient of the loss with respect to b. Dimension: (n_out, 1)
# dX: gradient of the loss with respect to X (for the previous layer). Dimension: (n_in, m)
# m: number of examples in the batch
# lr: learning rate, scalar
# dropout_rate: probability of dropout, scalar in [0, 1)


class Layer:
    """Base class for all layers."""
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError


class DenseLayer(Layer):
    """Fully connected layer with He initialization."""
    def __init__(self, input_size: int, output_size: int) -> None:
        self.dtype = np.float32
        scale = np.float32(np.sqrt(2.0 / input_size))
        self.W = np.random.randn(output_size, input_size).astype(self.dtype) * scale
        self.b = np.zeros((output_size, 1), dtype=self.dtype)
        self.X: np.ndarray | None = None
        self.Z: np.ndarray | None = None
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None
        self.dZ: np.ndarray | None = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute Z = WX + b."""
        X = X.astype(self.dtype, copy=False)
        self.X = X
        self.Z = self.W @ X + self.b
        return self.Z  # linear output used by activation layer

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        """Backpropagate gradient through linear transformation."""
        dZ = dZ.astype(self.dtype, copy=False)
        self.dZ = dZ
        m = self.X.shape[1]
        inv_m = np.float32(1.0 / m)
        self.dW = inv_m * (dZ @ self.X.T)
        self.db = inv_m * np.sum(dZ, axis=1, keepdims=True)
        dX = self.W.T @ dZ
        return dX


class ReLULayer(Layer):
    """Applies the ReLU activation element-wise."""
    def __init__(self) -> None:
        self.Z: np.ndarray | None = None
        self.A: np.ndarray | None = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        self.Z = Z
        self.A = ReLU(Z)
        return self.A

    def backward(self, dA: np.ndarray) -> np.ndarray:
        return dA * ReLU_deriv(self.Z)


class DropoutLayer(Layer):
    """Implements dropout regularization for dense activations."""
    def __init__(self, dropout_rate: float) -> None:
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError('dropout_rate must be in [0, 1).')
        self.dropout_rate = dropout_rate
        self.keep_prob = 1.0 - dropout_rate
        self.mask: np.ndarray | None = None
        self.A: np.ndarray | None = None

    def forward(self, A: np.ndarray, training: bool = True) -> np.ndarray:
        if self.dropout_rate == 0.0 or not training:
            self.mask = None
            self.A = A
            return self.A
        mask = (np.random.rand(*A.shape) < self.keep_prob).astype(A.dtype)
        self.mask = mask / np.float32(self.keep_prob)
        self.A = A * self.mask
        return self.A

    def backward(self, dA: np.ndarray) -> np.ndarray:
        if self.dropout_rate == 0.0 or self.mask is None:
            return dA
        return dA * self.mask


class SoftmaxLayer(Layer):
    """Applies the softmax activation to each column."""
    def __init__(self) -> None:
        self.Z: np.ndarray | None = None
        self.A: np.ndarray | None = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        self.Z = Z
        self.A = softmax(Z)
        return self.A

    def backward(self, Y: np.ndarray) -> np.ndarray:
        return (self.A - Y)


class CrossEntropyLoss:
    """Cross-entropy loss for multi-class classification."""
    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = np.float32(eps)
        self.A: np.ndarray | None = None
        self.Y: np.ndarray | None = None

    def forward(self, A: np.ndarray, Y: np.ndarray) -> float:
        m = Y.shape[1]
        A = A.astype(np.float32, copy=False)
        Y = Y.astype(np.float32, copy=False)
        A_clipped = np.clip(A, self.eps, np.float32(1.0) - self.eps)
        self.A, self.Y = A_clipped, Y
        return float(-np.sum(Y * np.log(A_clipped)) / m)

    def backward(self) -> np.ndarray:
        return self.A - self.Y