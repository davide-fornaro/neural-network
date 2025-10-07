"""Command-line entry point for training the feedforward neural network."""

import argparse
from typing import List
import numpy as np
from constants import (
    BATCH_SIZE,
    EPOCHS,
    HIDDEN_LAYERS_SIZES,
    LEARNING_RATE,
    DROPOUT_RATE,
    OPTIMIZER_CLS,
)
from dataset_downloader import DatasetDownloader
from neural_network import NeuralNetwork
from optimizer import Adam, AdamW, SGD, SGDMomentum

_SUPPORTED_DATASETS = ("mnist", "fashion_mnist", "cifar10")
_OPTIMIZER_MAP = {
    "sgd": SGD,
    "sgdmomentum": SGDMomentum,
    "adam": Adam,
    "adamw": AdamW,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the feedforward neural network on a vision dataset.",
    )
    parser.add_argument(
        "--dataset",
        default="mnist",
        choices=_SUPPORTED_DATASETS,
        help="Dataset to use (default: mnist).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Mini-batch size (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--optimizer",
        default=OPTIMIZER_CLS,
        choices=("SGD", "SGDMomentum", "Adam", "AdamW"),
        help=f"Optimizer to use (default: {OPTIMIZER_CLS}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE}).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DROPOUT_RATE,
        help=f"Dropout rate per gli strati nascosti (default: {DROPOUT_RATE}).",
    )
    parser.add_argument(
        "--hidden",
        nargs="+",
        type=int,
        default=None,
        help="Hidden layer sizes, e.g. --hidden 256 128.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Limit the number of training examples for quick experiments.",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="Limit the number of test examples for quick experiments.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Evaluate accuracy every N epochs (set to 0 to disable).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def _build_layer_sizes(input_size: int, output_size: int, hidden_sizes: List[int]) -> List[int]:
    return [input_size, *hidden_sizes, output_size]


def main() -> None:
    args = _parse_args()

    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("Dropout rate must be in [0, 1).")

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    dataset_name = args.dataset.lower()
    downloader = DatasetDownloader()
    downloader.ensure_dataset(dataset_name)
    input_size, output_size = downloader.dataset_info(dataset_name)

    hidden_sizes = list(args.hidden) if args.hidden else list(HIDDEN_LAYERS_SIZES)
    layer_sizes = _build_layer_sizes(input_size, output_size, hidden_sizes)

    optimizer_key = args.optimizer.lower()
    optimizer_cls = _OPTIMIZER_MAP[optimizer_key]
    network = NeuralNetwork(layer_sizes=layer_sizes, optimizer_cls=optimizer_cls, lr=args.lr, dropout_rate=args.dropout)

    X_train, Y_train, y_train = downloader.load(
        dataset_name,
        split="train",
        one_hot_labels=True,
        limit=args.train_limit,
    )
    if Y_train is None:
        raise RuntimeError("One-hot encoded labels are required for training.")

    X_test, _, y_test = downloader.load(
        dataset_name,
        split="test",
        one_hot_labels=False,
        limit=args.test_limit,
    )

    X_train = X_train.astype(np.float32, copy=False)
    Y_train = Y_train.astype(np.float32, copy=False)
    X_test = X_test.astype(np.float32, copy=False)
    y_train = y_train.astype(np.int64, copy=False)
    y_test = y_test.astype(np.int64, copy=False)

    num_train = X_train.shape[1]
    if num_train == 0:
        raise ValueError("Training set is empty. Adjust --train-limit.")

    print(
        f"Training on {dataset_name} with {num_train} samples, architecture: "
        + " -> ".join(str(size) for size in layer_sizes)
        + f", optimizer: {args.optimizer}, lr: {args.lr}, dropout: {args.dropout}"
    )

    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(num_train)
        epoch_loss = 0.0
        seen_samples = 0

        for start in range(0, num_train, args.batch_size):
            end = min(start + args.batch_size, num_train)
            idx = perm[start:end]
            loss = network.train_step(X_train[:, idx], Y_train[:, idx])
            batch_size = end - start
            epoch_loss += loss * batch_size
            seen_samples += batch_size

        epoch_loss /= max(seen_samples, 1)

        log_parts = [f"Epoch {epoch}/{args.epochs}", f"loss: {epoch_loss:.6f}"]

        if args.eval_interval > 0 and epoch % args.eval_interval == 0:
            train_acc = network.accuracy(X_train, y_train)
            test_acc = network.accuracy(X_test, y_test)
            log_parts.append(f"train_acc: {train_acc:.4f}")
            log_parts.append(f"test_acc: {test_acc:.4f}")

        print(" - ".join(log_parts))

    final_test_acc = network.accuracy(X_test, y_test)
    print(f"Final test accuracy: {final_test_acc:.4f}")


if __name__ == "__main__":
    main()
