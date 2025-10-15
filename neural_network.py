"""Feedforward neural network composed of Dense, ReLU, Dropout and Softmax layers."""

from typing import List, Type
import numpy as np
from layer import Layer, DenseLayer, DropoutLayer, ReLULayer, SoftmaxLayer, CrossEntropyLoss
from optimizer import SGD

class NeuralNetwork:
    """
    Simple fully-connected neural network.

    Args:
        layer_sizes: list of layer sizes, including input and output sizes. For example
            [n_in, h1, h2, ..., n_out] results in n_in → h1 → ReLU (and optional dropout)
            → h2 → ReLU → ... → n_out → Softmax.
        optimizer_cls: optimizer class to use (defaults to SGD).
        lr: learning rate.
        dropout_rate: dropout probability applied to hidden activations.
    """
    def __init__(
        self,
        layer_sizes: List[int],
        optimizer_cls: Type[object] = SGD,
        lr: float = 1e-1,
        dropout_rate: float = 0.0,
    ) -> None:
        self.layers: list[Layer] = []
        self.dropout_rate = dropout_rate

        num_transitions = len(layer_sizes) - 1
        for i in range(num_transitions):
            # add Dense layer
            self.layers.append(DenseLayer(layer_sizes[i], layer_sizes[i + 1]))
            # add activation (and optional dropout) for hidden layers
            is_hidden = i < num_transitions - 1
            if is_hidden:
                self.layers.append(ReLULayer())
                if dropout_rate > 0.0:
                    self.layers.append(DropoutLayer(dropout_rate))
        # finally append Softmax layer to output probabilities
        self.layers.append(SoftmaxLayer())

        # set optimizer to operate on trainable layers (Dense layers)
        self.optimizer = optimizer_cls(self.layers, lr=lr)
        self.criterion = CrossEntropyLoss()

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass through all layers."""
        A = X
        for layer in self.layers:
            if isinstance(layer, DropoutLayer):
                A = layer.forward(A, training=training)
            else:
                A = layer.forward(A)
        return A

    def compute_loss(self, A: np.ndarray, Y: np.ndarray) -> float:
        return self.criterion.forward(A, Y)

    def backward(self) -> None:
        """
        Initiates backpropagation from the loss. Starts by computing the gradient
        on the last layer using CrossEntropyLoss and Softmax, then propagates
        through preceding layers and updates weights using the optimizer.
        """
        # initial gradient from CrossEntropyLoss (uses saved predictions and labels)
        dZ = self.criterion.backward()
        # skip the final Softmax layer in backward propagation
        # self.layers[-1] is SoftmaxLayer
        assert isinstance(self.layers[-1], SoftmaxLayer)
        for layer in reversed(self.layers[:-1]):
            dZ = layer.backward(dZ)
        # update parameters of trainable layers
        self.optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        A = self.forward(X, training=False)
        return np.argmax(A, axis=0)

    def train_step(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Run a single training step (forward + backward)."""
        A = self.forward(X, training=True)
        loss = self.compute_loss(A, Y)
        self.backward()
        return loss

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X, training=False)

    def accuracy(self, X: np.ndarray, y_true: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == y_true))