import numpy as np


def ReLU(Z: np.ndarray) -> np.ndarray:
    """Compute the Rectified Linear Unit element-wise."""
    return np.maximum(Z, np.float32(0.0))


def ReLU_deriv(Z: np.ndarray) -> np.ndarray:
    """Derivative of the ReLU function."""
    return (Z > 0).astype(Z.dtype)


def softmax(Z: np.ndarray) -> np.ndarray:
    """Compute the column-wise softmax of the input matrix."""
    Z = Z.astype(np.float32, copy=False)
    Z_shift = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z_shift, dtype=np.float32)
    return expZ / np.sum(expZ, axis=0, keepdims=True, dtype=np.float32)


def one_hot(Y: np.ndarray) -> np.ndarray:
    """Convert label vector to one-hot encoded matrix."""
    m = Y.size
    K = int(Y.max()) + 1
    one_hot_Y = np.zeros((K, m), dtype=np.float32)
    one_hot_Y[Y, np.arange(m)] = np.float32(1.0)
    return one_hot_Y