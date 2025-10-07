"""Utilities for downloading and loading popular vision datasets.

This module downloads MNIST, Fashion-MNIST, and CIFAR-10 datasets and exposes a
helper class that returns data ready for the simple neural network defined in
this project.
"""
from __future__ import annotations

import gzip
import hashlib
import pickle
import struct
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np


_CHUNK_SIZE = 1 << 15


class DatasetDownloader:
    """Download and prepare vision datasets for the neural network."""

    _MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    _FASHION_BASE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    _CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    _MNIST_FILES = (
        {"filename": "train-images-idx3-ubyte.gz", "url": _MNIST_BASE_URL + "train-images-idx3-ubyte.gz", "md5": "f68b3c2dcbeaaa9fbdd348bbdeb94873"},
        {"filename": "train-labels-idx1-ubyte.gz", "url": _MNIST_BASE_URL + "train-labels-idx1-ubyte.gz", "md5": "d53e105ee54ea40749a09fcbcd1e9432"},
        {"filename": "t10k-images-idx3-ubyte.gz", "url": _MNIST_BASE_URL + "t10k-images-idx3-ubyte.gz", "md5": "9fb629c4189551a2d022fa330f9573f3"},
        {"filename": "t10k-labels-idx1-ubyte.gz", "url": _MNIST_BASE_URL + "t10k-labels-idx1-ubyte.gz", "md5": "ec29112dd5afa0611ce80d1b7f02629c"},
    )

    _FASHION_FILES = (
        {"filename": "train-images-idx3-ubyte.gz", "url": _FASHION_BASE_URL + "train-images-idx3-ubyte.gz", "md5": "8d4fb7e6c68d591d4c3dfef9ec88bf0d"},
        {"filename": "train-labels-idx1-ubyte.gz", "url": _FASHION_BASE_URL + "train-labels-idx1-ubyte.gz", "md5": "25c81989df183df01b3e8a0aad5dffbe"},
        {"filename": "t10k-images-idx3-ubyte.gz", "url": _FASHION_BASE_URL + "t10k-images-idx3-ubyte.gz", "md5": "bef4ecab320f06d8554ea6380940ec79"},
        {"filename": "t10k-labels-idx1-ubyte.gz", "url": _FASHION_BASE_URL + "t10k-labels-idx1-ubyte.gz", "md5": "bb300cfdad3c16e7a12a480ee83cd310"},
    )

    _CIFAR10_FILES = (
        {"filename": "cifar-10-python.tar.gz", "url": _CIFAR10_URL, "md5": "c58f30108f718f92721af3b95e74349a", "extract": "tar"},
    )

    _CONFIG: Dict[str, Dict[str, object]] = {
        "mnist": {
            "files": _MNIST_FILES,
            "type": "mnist_like",
            "num_classes": 10,
            "input_size": 784,
        },
        "fashion_mnist": {
            "files": _FASHION_FILES,
            "type": "mnist_like",
            "num_classes": 10,
            "input_size": 784,
        },
        "cifar10": {
            "files": _CIFAR10_FILES,
            "type": "cifar10",
            "num_classes": 10,
            "input_size": 3072,
        },
    }

    def __init__(self, data_dir: Union[Path, str] = "data") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def available_datasets(self) -> Tuple[str, ...]:
        return tuple(self._CONFIG.keys())

    def ensure_dataset(self, name: str) -> Path:
        name = name.lower()
        if name not in self._CONFIG:
            raise ValueError(f"Unsupported dataset '{name}'.")
        dataset_dir = self.data_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for file_info in self._CONFIG[name]["files"]:  # type: ignore[index]
            filename = file_info["filename"]  # type: ignore[index]
            url = file_info["url"]  # type: ignore[index]
            target = dataset_dir / filename
            expected_md5 = file_info.get("md5")  # type: ignore[union-attr]
            needs_download = True
            if target.exists():
                if expected_md5 and self._check_md5(target, str(expected_md5)):
                    needs_download = False
                elif expected_md5:
                    target.unlink()
                else:
                    needs_download = False
            if needs_download:
                print(f"Downloading {filename}...")
                self._download_file(url, target)
                if expected_md5 and not self._check_md5(target, str(expected_md5)):
                    target.unlink(missing_ok=True)
                    raise ValueError(f"Checksum mismatch for {filename}.")
            if file_info.get("extract") == "tar":  # type: ignore[union-attr]
                extracted_dir = dataset_dir / "cifar-10-batches-py"
                if not extracted_dir.exists():
                    self._extract_tar(target, dataset_dir)
        return dataset_dir

    def dataset_info(self, name: str) -> Tuple[int, int]:
        name = name.lower()
        if name not in self._CONFIG:
            raise ValueError(f"Unsupported dataset '{name}'.")
        info = self._CONFIG[name]
        return int(info["input_size"]), int(info["num_classes"])

    def load(
        self,
        name: str,
        split: str = "train",
        normalize: bool = True,
        one_hot_labels: bool = True,
        limit: Optional[int] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        split = split.lower()
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'.")
        dataset_dir = self.ensure_dataset(name)
        config = self._CONFIG[name.lower()]
        loader_type = config["type"]  # type: ignore[index]
        num_classes = int(config["num_classes"])  # type: ignore[index]
        if loader_type == "mnist_like":
            return self._load_mnist_like(dataset_dir, split, normalize, one_hot_labels, limit, num_classes)
        if loader_type == "cifar10":
            return self._load_cifar10(dataset_dir, split, normalize, one_hot_labels, limit, num_classes)
        raise ValueError(f"Unsupported dataset type '{loader_type}'.")

    def _download_file(self, url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = destination.with_suffix(destination.suffix + ".download")
        with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as fh:
            while True:
                chunk = response.read(_CHUNK_SIZE)
                if not chunk:
                    break
                fh.write(chunk)
        tmp_path.replace(destination)

    def _check_md5(self, path: Path, expected_md5: str) -> bool:
        hasher = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
                hasher.update(chunk)
        return hasher.hexdigest() == expected_md5

    def _extract_tar(self, archive_path: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        destination_resolved = destination.resolve()
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                member_path = (destination_resolved / member.name).resolve()
                if not self._is_within_directory(destination_resolved, member_path):
                    raise ValueError("Unsafe path detected during extraction.")
            tar.extractall(path=destination_resolved)

    def _is_within_directory(self, directory: Path, target: Path) -> bool:
        try:
            target.relative_to(directory)
            return True
        except ValueError:
            return False

    def _load_mnist_like(
        self,
        dataset_dir: Path,
        split: str,
        normalize: bool,
        one_hot_labels: bool,
        limit: Optional[int],
        num_classes: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        image_file = dataset_dir / ("train-images-idx3-ubyte.gz" if split == "train" else "t10k-images-idx3-ubyte.gz")
        label_file = dataset_dir / ("train-labels-idx1-ubyte.gz" if split == "train" else "t10k-labels-idx1-ubyte.gz")
        images = self._read_idx_images(image_file)
        labels = self._read_idx_labels(label_file)
        if limit is not None:
            images = images[:limit]
            labels = labels[:limit]
        X = images.astype(np.float32)
        if normalize:
            X /= 255.0
        X = X.T
        y = labels.astype(np.int64)
        Y = self._to_one_hot(y, num_classes) if one_hot_labels else None
        return X, Y, y

    def _load_cifar10(
        self,
        dataset_dir: Path,
        split: str,
        normalize: bool,
        one_hot_labels: bool,
        limit: Optional[int],
        num_classes: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        batches_dir = dataset_dir / "cifar-10-batches-py"
        if not batches_dir.exists():
            archive = dataset_dir / "cifar-10-python.tar.gz"
            if archive.exists():
                self._extract_tar(archive, dataset_dir)
            else:
                raise FileNotFoundError("CIFAR-10 archive missing. Re-run ensure_dataset().")
        batch_files: Iterable[str]
        if split == "train":
            batch_files = tuple(f"data_batch_{i}" for i in range(1, 6))
        else:
            batch_files = ("test_batch",)
        data_list = []
        label_list = []
        for batch_name in batch_files:
            batch_path = batches_dir / batch_name
            with open(batch_path, "rb") as fh:
                batch = pickle.load(fh, encoding="latin1")
            data_list.append(batch["data"])  # type: ignore[index]
            label_list.append(np.asarray(batch["labels"], dtype=np.int64))  # type: ignore[index]
        data = np.concatenate(data_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        if limit is not None:
            data = data[:limit]
            labels = labels[:limit]
        X = data.astype(np.float32)
        if normalize:
            X /= 255.0
        X = X.T
        y = labels.astype(np.int64)
        Y = self._to_one_hot(y, num_classes) if one_hot_labels else None
        return X, Y, y

    def _read_idx_images(self, path: Path) -> np.ndarray:
        with gzip.open(path, "rb") as fh:
            magic, num_images, rows, cols = struct.unpack(">IIII", fh.read(16))
            if magic != 2051:
                raise ValueError(f"Unexpected magic number {magic} in image file {path}.")
            data = fh.read(num_images * rows * cols)
        images = np.frombuffer(data, dtype=np.uint8)
        images = images.reshape(num_images, rows * cols)
        return images

    def _read_idx_labels(self, path: Path) -> np.ndarray:
        with gzip.open(path, "rb") as fh:
            magic, num_labels = struct.unpack(">II", fh.read(8))
            if magic != 2049:
                raise ValueError(f"Unexpected magic number {magic} in label file {path}.")
            data = fh.read(num_labels)
        labels = np.frombuffer(data, dtype=np.uint8)
        return labels

    def _to_one_hot(self, labels: np.ndarray, num_classes: int) -> np.ndarray:
        one_hot = np.zeros((num_classes, labels.size), dtype=np.float32)
        columns = np.arange(labels.size)
        one_hot[labels, columns] = 1.0
        return one_hot
