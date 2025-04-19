# app/utils/cnn_worker.py
"""QThread worker para entrenar y evaluar CNNs.

Usa las arquitecturas definidas en `app.models.cnn_models` y realiza un flujo
mínimo: cargar imágenes desde subcarpetas (una carpeta por clase), crear un
dataset `tf.data`, entrenar, evaluar y emitir logs a la GUI.
"""

from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
import tensorflow as tf
from sklearn.metrics import accuracy_score

from app.models.cnn_models import build_model


class CNNClassifierWorker(QThread):
    """Hilo que entrena y evalúa un modelo CNN."""

    log_signal = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        folder_path: str,
        model_id: str,
        input_size: int,
        feat_size: int,
        epochs: int = 10,
        batch_size: int = 8,
    ) -> None:
        super().__init__()
        self.folder_path = Path(folder_path)
        self.model_id = model_id
        self.input_size = input_size
        self.feat_size = feat_size
        self.epochs = epochs
        self.batch_size = batch_size

    # -------------------------------------------------------------
    def run(self):
        try:
            class_names = sorted([d.name for d in self.folder_path.iterdir() if d.is_dir()])
            num_classes = len(class_names)
            if num_classes == 0:
                raise RuntimeError("No se encontraron subcarpetas de clases en el dataset.")
            class_to_idx = {name: i for i, name in enumerate(class_names)}

            # Dataset loader -------------------------------------------------
            def _load(path):
                label_name = tf.strings.split(path, os.sep)[-2]
                label = tf.cast(class_to_idx[label_name.numpy().decode()], tf.int64)
                img = tf.io.read_file(path)
                img = tf.io.decode_image(img, channels=3, expand_animations=False)
                img = tf.image.resize(img, (self.input_size, self.input_size))
                img = tf.cast(img, tf.float32) / 255.0
                return img, label

            list_ds = tf.data.Dataset.list_files(str(self.folder_path / "*/*"), shuffle=True)
            full_ds = list_ds.map(lambda p: tf.py_function(_load, [p], [tf.float32, tf.int64]),
                                   num_parallel_calls=tf.data.AUTOTUNE)
            ds_size = full_ds.cardinality().numpy() if tf.executing_eagerly() else None
            train_size = int(0.8 * ds_size) if ds_size else None
            train_ds = full_ds.take(train_size).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            val_ds = full_ds.skip(train_size).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

            # Modelo --------------------------------------------------------
            model = build_model(self.model_id, self.input_size, num_classes, self.feat_size)
            self.log_signal.emit(model.summary(print_fn=lambda x: self.log_signal.emit(x)))

            # Entrenamiento --------------------------------------------------
            self.log_signal.emit("Entrenando CNN...")
            history = model.fit(train_ds, validation_data=val_ds, epochs=self.epochs, verbose=0)
            self.log_signal.emit("Entrenamiento terminado.")

            # Evaluación simple ---------------------------------------------
            y_true, y_pred = [], []
            for imgs, labels in val_ds:
                preds = tf.argmax(model(imgs, training=False), axis=1).numpy()
                y_true.extend(labels.numpy())
                y_pred.extend(preds)
            acc = accuracy_score(y_true, y_pred)
            self.log_signal.emit(f"Precisión en validación: {acc * 100:.2f}%")

        except Exception as e:
            self.log_signal.emit(f"Error en CNNWorker: {e}")
        finally:
            self.finished.emit()
