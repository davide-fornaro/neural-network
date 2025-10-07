from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets

from constants import (
    BATCH_SIZE,
    DROPOUT_RATE,
    EPOCHS,
    HIDDEN_LAYERS_SIZES,
    LEARNING_RATE,
)
from dataset_downloader import DatasetDownloader
from neural_network import NeuralNetwork
from optimizer import Adam, AdamW, SGD, SGDMomentum


_LABEL_MAPS: Dict[str, Sequence[str]] = {
    "mnist": tuple(str(i) for i in range(10)),
    "fashion_mnist": (
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ),
    "cifar10": (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ),
}

_IMAGE_SHAPES: Dict[str, Tuple[int, int, int]] = {
    "mnist": (28, 28, 1),
    "fashion_mnist": (28, 28, 1),
    "cifar10": (32, 32, 3),
}

_OPTIMIZER_MAP = {
    "SGD": SGD,
    "SGDMomentum": SGDMomentum,
    "Adam": Adam,
    "AdamW": AdamW,
}


@dataclass
class TrainingConfig:
    dataset: str
    epochs: int
    batch_size: int
    optimizer_name: str
    learning_rate: float
    dropout: float
    hidden_sizes: List[int]
    train_limit: Optional[int]
    test_limit: Optional[int]
    eval_interval: int
    seed: int


@dataclass
class PredictionEntry:
    index: int
    true_label: int
    predicted_label: int
    confidence: float
    image: np.ndarray


class TrainingWorker(QtCore.QObject):
    started = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)
    log_message = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int, int)
    network_blueprint = QtCore.pyqtSignal(dict)
    epoch_metrics = QtCore.pyqtSignal(int, float, float, float)
    predictions_ready = QtCore.pyqtSignal(list)
    dense_layer_stats = QtCore.pyqtSignal(list)

    def __init__(self, config: TrainingConfig, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.config = config
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    @QtCore.pyqtSlot()
    def run(self) -> None:
        self.started.emit()
        try:
            self._execute_training()
        except Exception as exc:  # pragma: no cover - safety net for GUI thread
            formatted = "\n".join(traceback.format_exception(exc.__class__, exc, exc.__traceback__))
            self.error.emit(formatted)
        finally:
            self.finished.emit()

    def _emit_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_message.emit(f"[{timestamp}] {message}")

    def _execute_training(self) -> None:
        config = self.config
        dataset_name = config.dataset.lower()
        label_names = _LABEL_MAPS.get(dataset_name)
        if label_names is None:
            raise ValueError(f"Unsupported dataset '{dataset_name}'.")
        image_shape = _IMAGE_SHAPES.get(dataset_name)
        if image_shape is None:
            raise ValueError(f"Unsupported dataset '{dataset_name}'.")

        rng = np.random.default_rng(config.seed)
        np.random.seed(config.seed)

        downloader = DatasetDownloader()
        self._emit_log(f"Verifica dataset '{dataset_name}'...")
        dataset_path = downloader.ensure_dataset(dataset_name)
        self._emit_log(f"Dataset pronto in {dataset_path}.")

        input_size, output_size = downloader.dataset_info(dataset_name)
        hidden_sizes = config.hidden_sizes or list(HIDDEN_LAYERS_SIZES)
        layer_sizes = [input_size, *hidden_sizes, output_size]

        optimizer_cls = _OPTIMIZER_MAP.get(config.optimizer_name, SGD)
        network = NeuralNetwork(
            layer_sizes=layer_sizes,
            optimizer_cls=optimizer_cls,
            lr=config.learning_rate,
            dropout_rate=config.dropout,
        )
        blueprint = {
            "input_size": input_size,
            "hidden_sizes": list(hidden_sizes),
            "output_size": output_size,
            "dropout": config.dropout,
        }
        self.network_blueprint.emit(blueprint)

        self._emit_log("Caricamento dataset train/test...")
        X_train, Y_train, y_train = downloader.load(
            dataset_name,
            split="train",
            one_hot_labels=True,
            limit=config.train_limit,
        )
        if Y_train is None:
            raise RuntimeError("La codifica one-hot delle etichette è obbligatoria per l'addestramento.")

        X_test, _, y_test = downloader.load(
            dataset_name,
            split="test",
            one_hot_labels=False,
            limit=config.test_limit,
        )

        X_train = X_train.astype(np.float32, copy=False)
        Y_train = Y_train.astype(np.float32, copy=False)
        X_test = X_test.astype(np.float32, copy=False)
        y_train = y_train.astype(np.int64, copy=False)
        y_test = y_test.astype(np.int64, copy=False)

        num_train = X_train.shape[1]
        if num_train == 0:
            raise ValueError("Il training set è vuoto. Riduci il limite del dataset o scegli un'altra configurazione.")

        epochs = max(1, config.epochs)
        batch_size = max(1, config.batch_size)
        eval_interval = max(1, config.eval_interval)

        num_predictions = min(100, X_test.shape[1])
        prediction_slice = slice(0, num_predictions)

        self._emit_log(
            "Inizio training: "
            + f"dataset={dataset_name}, epochs={epochs}, batch_size={batch_size}, optimizer={config.optimizer_name}, "
            + f"lr={config.learning_rate}, dropout={config.dropout}, eval_interval={eval_interval}"
        )

        dense_layers = [layer for layer in network.layers if hasattr(layer, "W") and hasattr(layer, "b")]

        for epoch in range(1, epochs + 1):
            if self._stop_requested:
                self._emit_log("Interruzione richiesta. Concludo l'addestramento...")
                break

            perm = rng.permutation(num_train)
            epoch_loss = 0.0
            seen_samples = 0

            for start in range(0, num_train, batch_size):
                if self._stop_requested:
                    break
                end = min(start + batch_size, num_train)
                idx = perm[start:end]
                loss = network.train_step(X_train[:, idx], Y_train[:, idx])
                current_batch = end - start
                epoch_loss += float(loss) * current_batch
                seen_samples += current_batch

            if seen_samples == 0:
                continue
            epoch_loss /= float(seen_samples)

            train_acc = network.accuracy(X_train, y_train)
            test_acc = network.accuracy(X_test, y_test)

            self.epoch_metrics.emit(epoch, float(epoch_loss), float(train_acc), float(test_acc))
            self.progress.emit(epoch, epochs)

            dense_stats_payload = []
            for layer in dense_layers:
                weight_norm = float(np.linalg.norm(layer.W))
                mean_abs_weight = float(np.mean(np.abs(layer.W)))
                bias_norm = float(np.linalg.norm(layer.b))
                dense_stats_payload.append(
                    {
                        "weight_norm": weight_norm,
                        "mean_abs_weight": mean_abs_weight,
                        "bias_norm": bias_norm,
                    }
                )
            if dense_stats_payload:
                self.dense_layer_stats.emit(dense_stats_payload)

            if num_predictions > 0 and (epoch % eval_interval == 0 or epoch == epochs):
                probs = network.predict_proba(X_test[:, prediction_slice])
                preds = np.argmax(probs, axis=0)
                true_labels = y_test[prediction_slice]
                confidences = probs[preds, np.arange(preds.size)]
                flat_samples = X_test[:, prediction_slice].T
                preview_images = self._prepare_preview_images(flat_samples, dataset_name, image_shape)

                prediction_entries = [
                    PredictionEntry(
                        index=i,
                        true_label=int(true),
                        predicted_label=int(pred),
                        confidence=float(conf),
                        image=preview_images[i],
                    )
                    for i, (true, pred, conf) in enumerate(zip(true_labels, preds, confidences))
                ]
                self.predictions_ready.emit(prediction_entries)

            self._emit_log(
                f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.6f} - train_acc: {train_acc:.4f} - test_acc: {test_acc:.4f}"
            )

        else:
            self._emit_log("Training completato.")

    def _prepare_preview_images(
        self,
        flat_samples: np.ndarray,
        dataset_name: str,
        image_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        if flat_samples.size == 0:
            return np.empty((0,) + image_shape[:2])

        height, width, channels = image_shape
        sample_count = flat_samples.shape[0]

        if dataset_name == "cifar10":
            reshaped = flat_samples.reshape(sample_count, channels, height, width)
            formatted = np.transpose(reshaped, (0, 2, 3, 1))
        else:
            formatted = flat_samples.reshape(sample_count, height, width)

        return np.ascontiguousarray(formatted)

class NetworkExplorerView(QtWidgets.QGraphicsView):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self.setBackgroundBrush(QtGui.QColor("#f8fafc"))
        self._dense_columns: List[List[QtWidgets.QGraphicsEllipseItem]] = []
        self._dense_base_brushes: List[QtGui.QBrush] = []
        self._column_spacing = 160.0
        self._operation_spacing = 120.0
        self.setMinimumHeight(260)

    def build_network(self, blueprint: Dict[str, object]) -> None:
        self._scene.clear()
        self.resetTransform()
        self._dense_columns.clear()
        self._dense_base_brushes.clear()

        input_size = int(blueprint.get("input_size", 0))
        hidden_sizes: Iterable[int] = blueprint.get("hidden_sizes", [])  # type: ignore[assignment]
        output_size = int(blueprint.get("output_size", 0))
        dropout_rate = float(blueprint.get("dropout", 0.0))

        element_rects: List[QtCore.QRectF] = []

        current_x = 0.0
        current_x, rect = self._add_node_column(
            current_x,
            input_size,
            label=f"Input\n{input_size} neuroni",
            color="#dbeafe",
            track_activity=False,
        )
        element_rects.append(rect)

        for idx, hidden_size in enumerate(hidden_sizes, start=1):
            current_x, rect = self._add_node_column(
                current_x,
                int(hidden_size),
                label=f"Dense {idx}\n{int(hidden_size)} neuroni",
                color="#bfdbfe",
                track_activity=True,
            )
            element_rects.append(rect)

            current_x, rect = self._add_operation_item(current_x, "ReLU", "#fef3c7")
            element_rects.append(rect)

            if dropout_rate > 0.0:
                current_x, rect = self._add_operation_item(
                    current_x,
                    f"Dropout\np={dropout_rate:.2f}",
                    "#fbcfe8",
                )
                element_rects.append(rect)

        current_x, rect = self._add_node_column(
            current_x,
            output_size,
            label=f"Output\n{output_size} classi",
            color="#c7d2fe",
            track_activity=True,
        )
        element_rects.append(rect)

        current_x, rect = self._add_operation_item(current_x, "Softmax", "#ede9fe")
        element_rects.append(rect)

        self._draw_connectors(element_rects)
        self._finalize_scene()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # type: ignore[override]
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        zoom_factor = 1.2 if delta > 0 else 1 / 1.2
        self.scale(zoom_factor, zoom_factor)
        event.accept()

    def update_dense_activity(self, stats: Sequence[Dict[str, float]]) -> None:
        if not self._dense_columns:
            return
        if not stats:
            self._reset_dense_colors()
            return
        max_mean = max((stat.get("mean_abs_weight", 0.0) for stat in stats), default=0.0) or 1.0
        for column_index, (column_items, stat) in enumerate(zip(self._dense_columns, stats)):
            intensity = min(stat.get("mean_abs_weight", 0.0) / max_mean, 1.0)
            color = QtGui.QColor.fromHsvF(0.58, max(0.15, intensity), 0.95)
            brush = QtGui.QBrush(color)
            for item in column_items:
                item.setBrush(brush)

        # Reset any remaining dense columns if stats length is shorter
        if len(stats) < len(self._dense_columns):
            for idx in range(len(stats), len(self._dense_columns)):
                base_brush = self._dense_base_brushes[idx]
                for item in self._dense_columns[idx]:
                    item.setBrush(QtGui.QBrush(base_brush))

    def _reset_dense_colors(self) -> None:
        for column_items, base_brush in zip(self._dense_columns, self._dense_base_brushes):
            for item in column_items:
                item.setBrush(QtGui.QBrush(base_brush))

    def _add_node_column(
        self,
        current_x: float,
        neuron_count: int,
        *,
        label: str,
        color: str,
        track_activity: bool,
    ) -> Tuple[float, QtCore.QRectF]:
        if neuron_count <= 0:
            return current_x + self._column_spacing, QtCore.QRectF()

        diameter, spacing = self._node_metrics(neuron_count)
        total_height = neuron_count * diameter + max(0, neuron_count - 1) * spacing
        top = -total_height / 2.0
        left = current_x - diameter / 2.0

        column_items: List[QtWidgets.QGraphicsEllipseItem] = []
        brush = QtGui.QBrush(QtGui.QColor(color))
        pen = QtGui.QPen(QtGui.QColor("#334155"), 0.6)

        for idx in range(neuron_count):
            ellipse = QtWidgets.QGraphicsEllipseItem(0.0, 0.0, diameter, diameter)
            ellipse.setPen(pen)
            ellipse.setBrush(QtGui.QBrush(brush))
            ellipse.setPos(left, top + idx * (diameter + spacing))
            tooltip_label = label.replace("\n", " ")
            ellipse.setToolTip(f"{tooltip_label} - neurone {idx + 1}")
            self._scene.addItem(ellipse)
            if track_activity:
                column_items.append(ellipse)

        self._add_label(current_x, top, diameter, label)

        if track_activity:
            self._dense_columns.append(column_items)
            self._dense_base_brushes.append(QtGui.QBrush(brush))

        rect = QtCore.QRectF(left, top, diameter, total_height)
        return current_x + self._column_spacing, rect

    def _add_operation_item(
        self,
        current_x: float,
        label: str,
        color: str,
    ) -> Tuple[float, QtCore.QRectF]:
        width = 120.0
        height = 70.0
        rect_item = QtWidgets.QGraphicsRectItem(-width / 2.0, -height / 2.0, width, height)
        rect_item.setBrush(QtGui.QBrush(QtGui.QColor(color)))
        rect_item.setPen(QtGui.QPen(QtGui.QColor("#4b5563"), 1.0))
        rect_item.setPos(current_x, 0.0)
        rect_item.setToolTip(label.replace("\n", " "))
        self._scene.addItem(rect_item)

        text_item = QtWidgets.QGraphicsTextItem(label)
        text_item.setDefaultTextColor(QtGui.QColor("#111827"))
        font = QtGui.QFont("Fira Code", 9)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        text_item.setFont(font)
        text_rect = text_item.boundingRect()
        text_item.setPos(current_x - text_rect.width() / 2.0, -text_rect.height() / 2.0)
        self._scene.addItem(text_item)

        rect = QtCore.QRectF(current_x - width / 2.0, -height / 2.0, width, height)
        return current_x + self._operation_spacing, rect

    def _add_label(self, center_x: float, top: float, diameter: float, label: str) -> None:
        text_item = QtWidgets.QGraphicsTextItem(label)
        text_item.setDefaultTextColor(QtGui.QColor("#111827"))
        font = QtGui.QFont("Fira Code", 9)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        text_item.setFont(font)
        text_rect = text_item.boundingRect()
        text_item.setPos(center_x - text_rect.width() / 2.0, top - text_rect.height() - 18.0)
        self._scene.addItem(text_item)

    def _node_metrics(self, neuron_count: int) -> Tuple[float, float]:
        if neuron_count <= 16:
            return 26.0, 16.0
        if neuron_count <= 64:
            return 18.0, 10.0
        if neuron_count <= 256:
            return 12.0, 6.0
        if neuron_count <= 1024:
            return 8.0, 4.0
        return 6.0, 3.0

    def _draw_connectors(self, rects: Sequence[QtCore.QRectF]) -> None:
        if len(rects) < 2:
            return
        for left_rect, right_rect in zip(rects, rects[1:]):
            if left_rect.isNull() or right_rect.isNull():
                continue
            line = QtWidgets.QGraphicsLineItem(
                left_rect.right(),
                0.0,
                right_rect.left(),
                0.0,
            )
            line.setPen(QtGui.QPen(QtGui.QColor("#94a3b8"), 1.4, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            self._scene.addItem(line)

    def _finalize_scene(self) -> None:
        bounds = self._scene.itemsBoundingRect()
        if bounds.isNull():
            return
        padded = bounds.marginsAdded(QtCore.QMarginsF(40.0, 40.0, 40.0, 40.0))
        self._scene.setSceneRect(padded)
        self.fitInView(self._scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

class MetricPlotWidget(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._figure = Figure(figsize=(6, 4), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        layout.addWidget(self._canvas)

        self._loss_ax = self._figure.add_subplot(2, 1, 1)
        self._acc_ax = self._figure.add_subplot(2, 1, 2)
        self._loss_ax.set_ylabel("Loss")
        self._acc_ax.set_ylabel("Accuracy")
        self._acc_ax.set_xlabel("Epoch")

        (self._loss_line,) = self._loss_ax.plot([], [], label="Loss", color="#2563eb")
        (self._train_acc_line,) = self._acc_ax.plot([], [], label="Train", color="#16a34a")
        (self._test_acc_line,) = self._acc_ax.plot([], [], label="Test", color="#dc2626")
        self._acc_ax.legend(loc="lower right")

        self._epochs: List[int] = []
        self._losses: List[float] = []
        self._train_accs: List[float] = []
        self._test_accs: List[float] = []

    def reset(self) -> None:
        self._epochs.clear()
        self._losses.clear()
        self._train_accs.clear()
        self._test_accs.clear()
        self._update_plot()

    def add_point(self, epoch: int, loss: float, train_acc: float, test_acc: float) -> None:
        self._epochs.append(epoch)
        self._losses.append(loss)
        self._train_accs.append(train_acc)
        self._test_accs.append(test_acc)
        self._update_plot()

    def _update_plot(self) -> None:
        self._loss_line.set_data(self._epochs, self._losses)
        self._train_acc_line.set_data(self._epochs, self._train_accs)
        self._test_acc_line.set_data(self._epochs, self._test_accs)

        if self._epochs:
            self._loss_ax.set_xlim(1, max(self._epochs))
            self._acc_ax.set_xlim(1, max(self._epochs))
            self._loss_ax.relim()
            self._loss_ax.autoscale_view()
            self._acc_ax.set_ylim(0.0, 1.0)
        else:
            self._loss_ax.set_xlim(0, 1)
            self._acc_ax.set_xlim(0, 1)
            self._acc_ax.set_ylim(0.0, 1.0)

        self._canvas.draw_idle()

class PredictionsGallery(QtWidgets.QScrollArea):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._container = QtWidgets.QWidget()
        self._layout = QtWidgets.QGridLayout(self._container)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setHorizontalSpacing(12)
        self._layout.setVerticalSpacing(12)
        self._layout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.setWidget(self._container)
        self._columns = 5
        self._cards: List[QtWidgets.QWidget] = []
        self._placeholder = QtWidgets.QLabel("Avvia il training per visualizzare le predizioni.")
        self._placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self._placeholder.setStyleSheet("color: #6b7280; font-style: italic;")
        self._layout.addWidget(self._placeholder, 0, 0, 1, self._columns)

    def clear(self) -> None:
        for card in self._cards:
            self._layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()
        self._show_placeholder()

    def update_predictions(
        self,
        entries: Sequence[PredictionEntry],
        label_names: Sequence[str],
    ) -> None:
        for card in self._cards:
            self._layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        if not entries:
            self._show_placeholder()
            return

        self._hide_placeholder()
        for idx, entry in enumerate(entries):
            card = self._create_card(entry, label_names)
            row = idx // self._columns
            col = idx % self._columns
            self._layout.addWidget(card, row, col)
            self._cards.append(card)

    def _show_placeholder(self) -> None:
        self._placeholder.show()

    def _hide_placeholder(self) -> None:
        self._placeholder.hide()

    def _create_card(
        self,
        entry: PredictionEntry,
        label_names: Sequence[str],
    ) -> QtWidgets.QWidget:
        card = QtWidgets.QFrame()
        card.setObjectName("predictionCard")
        card.setFrameShape(QtWidgets.QFrame.StyledPanel)
        correct = entry.true_label == entry.predicted_label
        background = "#dcfce7" if correct else "#fee2e2"
        border = "#16a34a" if correct else "#dc2626"
        card.setStyleSheet(
            "QFrame#predictionCard {"
            f" background-color: {background};"
            f" border: 1px solid {border};"
            " border-radius: 10px;"
            "}"
            "QLabel { color: #111827; }"
        )

        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title_label = QtWidgets.QLabel(f"Esempio #{entry.index}")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_label.setStyleSheet("font-weight: 600; color: #0f172a;")
        layout.addWidget(title_label)

        image_label = QtWidgets.QLabel()
        image_label.setAlignment(QtCore.Qt.AlignCenter)
        pixmap = self._image_to_pixmap(entry.image)
        if not pixmap.isNull():
            scaled = pixmap.scaled(112, 112, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            image_label.setPixmap(scaled)
        else:
            image_label.setText("N/D")
        layout.addWidget(image_label)

        pred_name = label_names[entry.predicted_label]
        true_name = label_names[entry.true_label]

        pred_label = QtWidgets.QLabel(f"Pred: <b>{pred_name}</b>")
        pred_label.setAlignment(QtCore.Qt.AlignCenter)
        pred_label.setTextFormat(QtCore.Qt.RichText)
        layout.addWidget(pred_label)

        true_label = QtWidgets.QLabel(f"Vero: {true_name}")
        true_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(true_label)

        confidence_label = QtWidgets.QLabel(f"Confidenza: {entry.confidence:.2%}")
        confidence_label.setAlignment(QtCore.Qt.AlignCenter)
        confidence_label.setStyleSheet("color: #374151;")
        layout.addWidget(confidence_label)

        card.setToolTip(
            f"Vero: {true_name}\nPredetto: {pred_name}\nConfidenza: {entry.confidence:.2%}"
        )

        return card

    def _image_to_pixmap(self, image: np.ndarray) -> QtGui.QPixmap:
        array = np.asarray(image)
        if array.size == 0:
            return QtGui.QPixmap()

        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]

        if array.ndim == 2:
            normalized = self._normalize_image(array)
            buffer = np.ascontiguousarray((normalized * 255.0).astype(np.uint8))
            qimage = QtGui.QImage(
                buffer.data,
                buffer.shape[1],
                buffer.shape[0],
                buffer.strides[0],
                QtGui.QImage.Format_Grayscale8,
            )
            return QtGui.QPixmap.fromImage(qimage.copy())

        if array.ndim == 3 and array.shape[2] == 3:
            normalized = self._normalize_image(array)
            buffer = np.ascontiguousarray((normalized * 255.0).astype(np.uint8))
            qimage = QtGui.QImage(
                buffer.data,
                buffer.shape[1],
                buffer.shape[0],
                buffer.strides[0],
                QtGui.QImage.Format_RGB888,
            )
            return QtGui.QPixmap.fromImage(qimage.copy())

        return QtGui.QPixmap()

    @staticmethod
    def _normalize_image(image: np.ndarray) -> np.ndarray:
        if image.size == 0:
            return image.astype(np.float32)
        min_val = float(np.min(image))
        max_val = float(np.max(image))
        if max_val - min_val < 1e-6:
            return np.zeros_like(image, dtype=np.float32)
        if max_val > 1.0 or min_val < 0.0:
            scaled = (image - min_val) / (max_val - min_val)
        else:
            scaled = np.clip(image, 0.0, 1.0)
        return scaled.astype(np.float32)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Neural Network Trainer")
        self.resize(1400, 900)

        self._worker_thread: Optional[QtCore.QThread] = None
        self._worker: Optional[TrainingWorker] = None
        self._current_labels: Sequence[str] = _LABEL_MAPS["mnist"]

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        self._config_group = QtWidgets.QGroupBox("Configurazione")
        self._config_layout = QtWidgets.QGridLayout(self._config_group)
        self._build_configuration_panel()
        main_layout.addWidget(self._config_group)

        controls_layout = QtWidgets.QHBoxLayout()
        self._start_button = QtWidgets.QPushButton("Avvia training")
        self._stop_button = QtWidgets.QPushButton("Ferma")
        self._stop_button.setEnabled(False)
        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        controls_layout.addWidget(self._start_button)
        controls_layout.addWidget(self._stop_button)
        controls_layout.addWidget(self._progress, 1)
        main_layout.addLayout(controls_layout)

        content_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        main_layout.addWidget(content_splitter, 1)

        upper_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        content_splitter.addWidget(upper_splitter)

        self._network_group = QtWidgets.QGroupBox("Architettura della rete")
        network_layout = QtWidgets.QVBoxLayout(self._network_group)
        self._network_view = NetworkExplorerView()
        network_layout.addWidget(self._network_view)
        upper_splitter.addWidget(self._network_group)

        self._metrics_group = QtWidgets.QGroupBox("Metriche di training")
        metrics_layout = QtWidgets.QVBoxLayout(self._metrics_group)
        self._metrics_plot = MetricPlotWidget()
        metrics_layout.addWidget(self._metrics_plot)
        upper_splitter.addWidget(self._metrics_group)

        self._predictions_group = QtWidgets.QGroupBox("Galleria predizioni (primi 100 esempi di test)")
        preds_layout = QtWidgets.QVBoxLayout(self._predictions_group)
        self._predictions_gallery = PredictionsGallery()
        preds_layout.addWidget(self._predictions_gallery)
        upper_splitter.addWidget(self._predictions_group)

        self._log_group = QtWidgets.QGroupBox("Log")
        log_layout = QtWidgets.QVBoxLayout(self._log_group)
        self._log_view = QtWidgets.QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(5000)
        log_layout.addWidget(self._log_view)
        content_splitter.addWidget(self._log_group)
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 1)

        self.statusBar().showMessage("Pronto")

        self._start_button.clicked.connect(self._start_training)
        self._stop_button.clicked.connect(self._stop_training)

    def _build_configuration_panel(self) -> None:
        row = 0

        dataset_label = QtWidgets.QLabel("Dataset")
        self._dataset_combo = QtWidgets.QComboBox()
        for key, name in (
            ("mnist", "MNIST"),
            ("fashion_mnist", "Fashion-MNIST"),
            ("cifar10", "CIFAR-10"),
        ):
            self._dataset_combo.addItem(name, key)
        self._dataset_combo.currentIndexChanged.connect(self._on_dataset_changed)
        self._config_layout.addWidget(dataset_label, row, 0)
        self._config_layout.addWidget(self._dataset_combo, row, 1)

        optimizer_label = QtWidgets.QLabel("Ottimizzatore")
        self._optimizer_combo = QtWidgets.QComboBox()
        for name in ("SGD", "SGDMomentum", "Adam", "AdamW"):
            self._optimizer_combo.addItem(name, name)
        row += 1
        self._config_layout.addWidget(optimizer_label, row, 0)
        self._config_layout.addWidget(self._optimizer_combo, row, 1)

        epochs_label = QtWidgets.QLabel("Epochs")
        self._epochs_spin = QtWidgets.QSpinBox()
        self._epochs_spin.setRange(1, 10000)
        self._epochs_spin.setValue(EPOCHS)
        row += 1
        self._config_layout.addWidget(epochs_label, row, 0)
        self._config_layout.addWidget(self._epochs_spin, row, 1)

        batch_label = QtWidgets.QLabel("Batch size")
        self._batch_spin = QtWidgets.QSpinBox()
        self._batch_spin.setRange(1, 2048)
        self._batch_spin.setValue(BATCH_SIZE)
        row += 1
        self._config_layout.addWidget(batch_label, row, 0)
        self._config_layout.addWidget(self._batch_spin, row, 1)

        lr_label = QtWidgets.QLabel("Learning rate")
        self._lr_spin = QtWidgets.QDoubleSpinBox()
        self._lr_spin.setDecimals(5)
        self._lr_spin.setSingleStep(0.0001)
        self._lr_spin.setRange(1e-5, 1.0)
        self._lr_spin.setValue(LEARNING_RATE)
        row += 1
        self._config_layout.addWidget(lr_label, row, 0)
        self._config_layout.addWidget(self._lr_spin, row, 1)

        dropout_label = QtWidgets.QLabel("Dropout")
        self._dropout_spin = QtWidgets.QDoubleSpinBox()
        self._dropout_spin.setDecimals(2)
        self._dropout_spin.setRange(0.0, 0.9)
        self._dropout_spin.setSingleStep(0.05)
        self._dropout_spin.setValue(DROPOUT_RATE)
        row += 1
        self._config_layout.addWidget(dropout_label, row, 0)
        self._config_layout.addWidget(self._dropout_spin, row, 1)

        hidden_label = QtWidgets.QLabel("Hidden layers")
        self._hidden_edit = QtWidgets.QLineEdit(",".join(str(s) for s in HIDDEN_LAYERS_SIZES))
        self._hidden_edit.setPlaceholderText("Esempio: 256,128,64")
        row += 1
        self._config_layout.addWidget(hidden_label, row, 0)
        self._config_layout.addWidget(self._hidden_edit, row, 1)

        train_limit_label = QtWidgets.QLabel("Limite train")
        self._train_limit_spin = QtWidgets.QSpinBox()
        self._train_limit_spin.setRange(0, 60000)
        self._train_limit_spin.setSpecialValueText("Tutti")
        row += 1
        self._config_layout.addWidget(train_limit_label, row, 0)
        self._config_layout.addWidget(self._train_limit_spin, row, 1)

        test_limit_label = QtWidgets.QLabel("Limite test")
        self._test_limit_spin = QtWidgets.QSpinBox()
        self._test_limit_spin.setRange(0, 10000)
        self._test_limit_spin.setSpecialValueText("Tutti")
        row += 1
        self._config_layout.addWidget(test_limit_label, row, 0)
        self._config_layout.addWidget(self._test_limit_spin, row, 1)

        eval_label = QtWidgets.QLabel("Eval interval")
        self._eval_spin = QtWidgets.QSpinBox()
        self._eval_spin.setRange(1, 1000)
        self._eval_spin.setValue(1)
        row += 1
        self._config_layout.addWidget(eval_label, row, 0)
        self._config_layout.addWidget(self._eval_spin, row, 1)

        seed_label = QtWidgets.QLabel("Seed")
        self._seed_spin = QtWidgets.QSpinBox()
        self._seed_spin.setRange(0, 1_000_000)
        self._seed_spin.setValue(42)
        row += 1
        self._config_layout.addWidget(seed_label, row, 0)
        self._config_layout.addWidget(self._seed_spin, row, 1)

        self._config_layout.setColumnStretch(1, 1)

    def _on_dataset_changed(self, index: int) -> None:
        dataset_key = self._dataset_combo.itemData(index)
        if not dataset_key:
            return
        self._current_labels = _LABEL_MAPS.get(dataset_key, _LABEL_MAPS["mnist"])
        self._predictions_gallery.clear()
        self._log(f"Dataset selezionato: {dataset_key}")

    def _collect_config(self) -> TrainingConfig:
        hidden_text = self._hidden_edit.text().strip()
        hidden_sizes = [int(value) for value in hidden_text.split(",") if value.strip().isdigit()]
        train_limit = self._train_limit_spin.value() or None
        test_limit = self._test_limit_spin.value() or None

        return TrainingConfig(
            dataset=str(self._dataset_combo.currentData()),
            epochs=self._epochs_spin.value(),
            batch_size=self._batch_spin.value(),
            optimizer_name=str(self._optimizer_combo.currentData()),
            learning_rate=self._lr_spin.value(),
            dropout=self._dropout_spin.value(),
            hidden_sizes=hidden_sizes,
            train_limit=train_limit,
            test_limit=test_limit,
            eval_interval=self._eval_spin.value(),
            seed=self._seed_spin.value(),
        )

    def _start_training(self) -> None:
        if self._worker_thread is not None:
            return
        config = self._collect_config()
        self._metrics_plot.reset()
        self._progress.setRange(0, config.epochs)
        self._progress.setValue(0)
        self._predictions_gallery.clear()

        self._worker = TrainingWorker(config)
        self._worker_thread = QtCore.QThread(self)
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._thread_finished)

        self._worker.log_message.connect(self._log)
        self._worker.error.connect(self._handle_error)
        self._worker.progress.connect(self._update_progress)
        self._worker.network_blueprint.connect(self._network_view.build_network)
        self._worker.epoch_metrics.connect(self._update_metrics)
        self._worker.predictions_ready.connect(self._update_predictions)
        self._worker.dense_layer_stats.connect(self._network_view.update_dense_activity)

        self._set_controls_enabled(False)
        self.statusBar().showMessage("Training in esecuzione...")
        self._worker_thread.start()

    def _stop_training(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._log("Interruzione del training richiesta dall'utente.")

    def _thread_finished(self) -> None:
        self._worker_thread.deleteLater()
        self._worker_thread = None
        self._worker = None
        self._set_controls_enabled(True)
        self.statusBar().showMessage("Pronto")
        self._progress.setRange(0, 1)
        self._progress.setValue(0)

    def _update_progress(self, current: int, total: int) -> None:
        self._progress.setMaximum(total)
        self._progress.setValue(current)

    def _update_metrics(self, epoch: int, loss: float, train_acc: float, test_acc: float) -> None:
        self._metrics_plot.add_point(epoch, loss, train_acc, test_acc)

    def _update_predictions(self, entries: Sequence[PredictionEntry]) -> None:
        self._predictions_gallery.update_predictions(entries, self._current_labels)

    def _set_controls_enabled(self, enabled: bool) -> None:
        self._config_group.setEnabled(enabled)
        self._start_button.setEnabled(enabled)
        self._stop_button.setEnabled(not enabled)

    def _handle_error(self, message: str) -> None:
        self._log(message)
        QtWidgets.QMessageBox.critical(self, "Errore", message)

    def _log(self, message: str) -> None:
        self._log_view.appendPlainText(message)
        self._log_view.verticalScrollBar().setValue(self._log_view.verticalScrollBar().maximum())


def main() -> None:
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
